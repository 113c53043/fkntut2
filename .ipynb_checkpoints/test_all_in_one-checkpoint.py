import os
import sys
import torch
import numpy as np
import cv2
import lpips
import json
import random
import shutil
import subprocess
import re
import gc
from collections import defaultdict
from tqdm import tqdm
from PIL import Image
from omegaconf import OmegaConf
from torch import autocast

# === 嘗試導入評估庫 ===
try:
    from pytorch_fid import fid_score
    from piq import brisque
    PIQ_AVAILABLE = True
except ImportError:
    print("⚠️ 缺少 piq 或 pytorch-fid 庫，部分畫質指標可能無法計算。")
    PIQ_AVAILABLE = False

# === 路徑設定 (請根據您的環境修改) ===
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

# 依賴模組
try:
    from ldm.util import instantiate_from_config
    from ldm.models.diffusion.dpm_solver import DPMSolverSampler
    from pure_alice_dynamic import generate_alice_image # 導入動態 Alice
    from robust_eval import identity, jpeg, resize, gblur, awgn # 導入攻擊
    from utils import load_512
except ImportError as e:
    print(f"❌ 導入失敗: {e}")
    print("請確保 pure_alice_dynamic.py, robust_eval.py, utils.py 都在同一目錄或路徑下。")
    sys.exit(1)

# 模型與數據集路徑
MAS_GRDH_PATH = PARENT_DIR
CKPT_PATH = "/home/vcpuser/netdrive/Workspace/stt/mas_GRDH/weights/v1-5-pruned.ckpt" 
if not os.path.exists(CKPT_PATH): # Fallback
    CKPT_PATH = os.path.join(MAS_GRDH_PATH, "weights/v1-5-pruned.ckpt")
CONFIG_PATH = os.path.join(MAS_GRDH_PATH, "configs/stable-diffusion/ldm.yaml")
DIR_REAL_COCO = os.path.join(CURRENT_DIR, "coco_val2017") # 真實 COCO 圖片路徑
PATH_CAPTIONS = os.path.join(CURRENT_DIR, "coco_annotations", "captions_val2017.json")

# 輸出路徑
OUTPUT_ROOT = os.path.join(MAS_GRDH_PATH, "outputs", "all_in_one_test")
DIR_COVER = os.path.join(OUTPUT_ROOT, "cover")
DIR_OURS = os.path.join(OUTPUT_ROOT, "ours")
DIR_REAL_RESIZED = os.path.join(OUTPUT_ROOT, "real_resized")
DIR_ATTACKED = os.path.join(OUTPUT_ROOT, "attacked_temp")

# 測試數量
TEST_SAMPLES = 100

# === 輔助函數 ===

def load_model():
    print(f"⏳ Loading SD Model...")
    config = OmegaConf.load(CONFIG_PATH)
    # Fix config
    def recursive_fix(conf):
        if isinstance(conf, (dict, OmegaConf)):
            for key in conf.keys():
                if key == "image_size" and conf[key] == 32: conf[key] = 64
                recursive_fix(conf[key])
    recursive_fix(config.model)
    
    pl_sd = torch.load(CKPT_PATH, map_location="cpu")
    sd = pl_sd["state_dict"] if "state_dict" in pl_sd else pl_sd
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model

def generate_cover(model, sampler, prompt, out_path, seed):
    if os.path.exists(out_path): return
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    c = model.get_learned_conditioning([prompt])
    uc = model.get_learned_conditioning([""])
    shape = (4, 64, 64)
    x_T = torch.randn(1, *shape, device="cuda")
    with torch.no_grad(), autocast("cuda"):
        z_enc, _ = sampler.sample(steps=20, conditioning=c, batch_size=1, shape=shape,
                                  unconditional_guidance_scale=5.0, unconditional_conditioning=uc,
                                  x_T=x_T, verbose=False)
        x_samples = model.decode_first_stage(z_enc)
    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
    Image.fromarray((x_samples[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)).save(out_path)

def load_prompts(limit):
    with open(PATH_CAPTIONS, 'r') as f:
        data = json.load(f)
    captions = [item['caption'] for item in data['annotations']]
    random.shuffle(captions)
    return captions[:limit]

def resize_real_images(src, dst, limit):
    if not os.path.exists(src): return
    os.makedirs(dst, exist_ok=True)
    files = [f for f in os.listdir(src) if f.lower().endswith(('.jpg', '.png'))][:limit]
    if len(os.listdir(dst)) >= len(files): return
    print("⚙️  Resizing real images for FID...")
    for f in tqdm(files):
        try:
            Image.open(os.path.join(src, f)).convert('RGB').resize((512,512)).save(os.path.join(dst, f))
        except: pass

def get_brisque(path):
    if not PIQ_AVAILABLE: return 0.0
    try:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        t_img = torch.tensor(img).permute(2,0,1).unsqueeze(0).float().cuda()
        return brisque(t_img, data_range=1.0).item()
    except: return 0.0

def run_bob_decode(img_path, prompt, key, gt_path):
    # 調用 pure_bob.py 進行解碼
    cmd = [
        sys.executable, "pure_bob.py",
        "--img_path", img_path,
        "--prompt", prompt,
        "--secret_key", str(key),
        "--gt_path", gt_path,
        "--ckpt", CKPT_PATH,
        "--config", CONFIG_PATH,
        "--dpm_steps", "20"
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        match = re.search(r"Bit Accuracy.*?: (\d+\.\d+)%", res.stdout)
        return float(match.group(1)) if match else 0.0
    except: return 0.0

# === 主流程 ===
def main():
    print(f"🚀 All-in-One Test: {TEST_SAMPLES} Samples 🚀")
    print(f"Testing: Quality (FID/BRISQUE) & Robustness (Acc)")
    
    # Init Dirs
    for d in [DIR_COVER, DIR_OURS, DIR_REAL_RESIZED, DIR_ATTACKED]:
        os.makedirs(d, exist_ok=True)
        
    # Payload
    payload_path = os.path.join(OUTPUT_ROOT, "payload.dat")
    with open(payload_path, "wb") as f: f.write(os.urandom(2048))
    with open(payload_path, "rb") as f: 
        raw_data = f.read()
        CAPACITY_BYTES = 16384 // 8 
        if len(raw_data) > CAPACITY_BYTES - 2: raw_data = raw_data[:CAPACITY_BYTES-2]
        final_payload = len(raw_data).to_bytes(2, 'big') + raw_data + b'\x00' * (CAPACITY_BYTES - len(raw_data) - 2)

    # 1. 生成階段
    print("\n[Phase 1] Generating Images...")
    prompts = load_prompts(TEST_SAMPLES)
    model = load_model()
    sampler = DPMSolverSampler(model)
    
    lpips_fn = lpips.LPIPS(net='alex').cuda()
    
    records = [] # Store metadata for robustness test
    
    for i, p in enumerate(tqdm(prompts)):
        key = 10000 + i
        p_cover = os.path.join(DIR_COVER, f"{i:05d}.png")
        p_ours = os.path.join(DIR_OURS, f"{i:05d}.png")
        gt_bits_path = p_ours + ".gt_bits.npy"
        
        # Save GT bits for Bob
        np.save(gt_bits_path, np.frombuffer(final_payload, dtype=np.uint8))
        
        # Generate
        generate_cover(model, sampler, p, p_cover, key)
        if not os.path.exists(p_ours):
            generate_alice_image(model, sampler, p, key, final_payload, p_ours, opt_iters=15, use_uncertainty=True)
            
        records.append({"id": i, "prompt": p, "key": key, "cover": p_cover, "ours": p_ours, "gt": gt_bits_path})
        
    del model, sampler
    torch.cuda.empty_cache()
    gc.collect()
    
    # 2. 畫質評估
    print("\n[Phase 2] Evaluating Quality...")
    brisque_scores = []
    lpips_scores = []
    
    for r in tqdm(records):
        brisque_scores.append(get_brisque(r["ours"]))
        
        # LPIPS
        img_c = load_512(r["cover"]).cuda()
        img_o = load_512(r["ours"]).cuda()
        with torch.no_grad():
            lpips_scores.append(lpips_fn(img_c, img_o).item())
            
    # FID
    resize_real_images(DIR_REAL_COCO, DIR_REAL_RESIZED, TEST_SAMPLES)
    try:
        fid_val = fid_score.calculate_fid_given_paths([DIR_REAL_RESIZED, DIR_OURS], 50, "cuda", 2048, 0)
    except: fid_val = 999.0
    
    # 3. 魯棒性評估
    print("\n[Phase 3] Evaluating Robustness...")
    attacks = [
        ("Identity", identity, None),
        ("JPEG(80)", jpeg, 80),
        ("Resize(0.5)", resize, 0.5),
        ("GBlur(3x3)", gblur, 3),
        ("AWGN(0.01)", awgn, 0.01)
    ]
    
    robust_results = defaultdict(list)
    
    for name, attack_fn, param in attacks:
        print(f"  Testing {name}...")
        for r in tqdm(records):
            # Apply Attack
            img = load_512(r["ours"]).cuda()
            tmp_name = os.path.join(DIR_ATTACKED, f"temp_{r['id']}")
            attack_fn(img, param, tmp_image_name=tmp_name)
            
            # Find output file (handling extension changes by attack)
            attacked_file = tmp_name + (".jpg" if "JPEG" in name else ".png")
            
            # Decode
            acc = run_bob_decode(attacked_file, r["prompt"], r["key"], r["gt"])
            robust_results[name].append(acc)
            
    # === 最終報告 ===
    print("\n" + "="*60)
    print(f"📊 FINAL REPORT (Samples: {TEST_SAMPLES})")
    print("="*60)
    print(f"Quality Metrics:")
    print(f"  FID (vs COCO)   : {fid_val:.4f} (Note: 100 samples is low for FID)")
    print(f"  BRISQUE (Ours)  : {np.mean(brisque_scores):.4f}")
    print(f"  LPIPS (vs Cover): {np.mean(lpips_scores):.4f}")
    print("-" * 60)
    print(f"Robustness (Bit Accuracy):")
    for name in attacks:
        acc_avg = np.mean(robust_results[name[0]])
        print(f"  {name[0]:<15} : {acc_avg:.2f}%")
    print("="*60)

if __name__ == "__main__":
    main()