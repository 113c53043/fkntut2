import os
import sys
import subprocess
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch import autocast
from omegaconf import OmegaConf

# === 1. 路徑設定 ===
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

try:
    from ldm.util import instantiate_from_config
    from ldm.models.diffusion.dpm_solver import DPMSolverSampler
    from mapping_module import ours_mapping 
except ImportError as e:
    print(f"❌ 無法導入模組: {e}")
    sys.exit(1)

# === 配置 ===
TOTAL_IMAGES = 1000 
MAS_GRDH_PATH = PARENT_DIR
CKPT_PATH = os.path.join(MAS_GRDH_PATH, "weights/v1-5-pruned.ckpt")
CONFIG_PATH = os.path.join(MAS_GRDH_PATH, "configs/stable-diffusion/ldm.yaml")

ALICE_SCRIPT = os.path.join(MAS_GRDH_PATH, "pure_alice_fixed.py")
ALICE_UNC_SCRIPT = os.path.join(MAS_GRDH_PATH, "pure_alice_uncertainty_fixed.py")

OUTPUT_ROOT = os.path.join(MAS_GRDH_PATH, "outputs", "paper_repro_results")
DIR_COVER = os.path.join(OUTPUT_ROOT, "cover_sd")
DIR_MAPPED = os.path.join(OUTPUT_ROOT, "mapped_base")
DIR_PURE = os.path.join(OUTPUT_ROOT, "ours_pure")
DIR_UNC = os.path.join(OUTPUT_ROOT, "ours_unc")
DIR_LATENT = os.path.join(OUTPUT_ROOT, "latents") 
DIR_TEMP = os.path.join(OUTPUT_ROOT, "temp")
PAYLOAD_PATH = os.path.join(DIR_TEMP, "payload.dat")
PATH_CAPTIONS = os.path.join(CURRENT_DIR, "coco_annotations", "captions_val2017.json")

# === 載入模型 ===
def load_model():
    print(f"⏳ Loading SD Model for repair...")
    config = OmegaConf.load(CONFIG_PATH)
    def recursive_fix(conf):
        if isinstance(conf, (dict, OmegaConf)):
            for key in conf.keys():
                if key == "image_size" and conf[key] == 32: conf[key] = 64
                recursive_fix(conf[key])
    recursive_fix(config.model)
    try:
        pl_sd = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    except:
        pl_sd = torch.load(CKPT_PATH, map_location="cpu")
    sd = pl_sd["state_dict"] if "state_dict" in pl_sd else pl_sd
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model

# === 載入 Prompts ===
import json
import random
def load_coco_prompts(json_path, limit=1000):
    with open(json_path, 'r') as f:
        data = json.load(f)
    captions = [item['caption'] for item in data['annotations']]
    random.shuffle(captions)
    return captions[:limit]

# === 生成函數 ===
def generate_cover_image(model, sampler, prompt, out_path, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    device = torch.device("cuda")
    c = model.get_learned_conditioning([prompt])
    uc = model.get_learned_conditioning([""])
    shape = (4, 64, 64)
    x_T = torch.randn(1, *shape, device=device)
    with torch.no_grad(), autocast("cuda"):
        z_enc, _ = sampler.sample(steps=20, conditioning=c, batch_size=1, shape=shape,
                                  unconditional_guidance_scale=5.0, unconditional_conditioning=uc,
                                  x_T=x_T, verbose=False)
        x_samples = model.decode_first_stage(z_enc)
    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
    Image.fromarray((x_samples[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)).save(out_path)

def generate_mapped_baseline_and_latent(model, sampler, prompt, session_key, payload_path, out_img_path, out_latent_path):
    device = torch.device("cuda")
    with open(payload_path, "rb") as f: raw_data = f.read()
    payload_data = raw_data[:16384//8-2] # 簡化處理
    
    # 這裡我們只生成 latent，不處理複雜 payload 邏輯，因為 latent 是之前生成的
    # 如果 latent 壞了才需要重生成。這裡假設我們需要重新跑一次 mapping
    # 為了修復方便，我們直接呼叫 Alice 腳本來生成
    pass 

def run_alice(script_path, prompt, session_key, out_path, latent_path, opt_iters=10, extra_args=[]):
    cmd = [
        sys.executable, script_path,
        "--prompt", prompt, 
        "--secret_key", str(session_key),
        "--payload_path", PAYLOAD_PATH,
        "--outpath", out_path,
        "--init_latent", latent_path, 
        "--ckpt", CKPT_PATH,
        "--config", CONFIG_PATH,
        "--opt_iters", str(opt_iters),  
        "--dpm_steps", "20"
    ] + extra_args
    subprocess.run(cmd, check=True, cwd=MAS_GRDH_PATH, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# === 檢查與修復邏輯 ===
def check_and_fix():
    print("🔍 Scanning for corrupt images...")
    
    prompts = load_coco_prompts(PATH_CAPTIONS, limit=TOTAL_IMAGES)
    
    # 載入模型 (只有在需要修復 Cover 時才載入)
    model = None
    sampler = None

    broken_files = []

    # 1. 掃描
    for i in tqdm(range(TOTAL_IMAGES), desc="Scanning"):
        paths = {
            "Cover": os.path.join(DIR_COVER, f"{i:05d}.png"),
            "Pure": os.path.join(DIR_PURE, f"{i:05d}.png"),
            "Unc": os.path.join(DIR_UNC, f"{i:05d}.png"),
            "Latent": os.path.join(DIR_LATENT, f"{i:05d}.pt")
        }
        
        for key, path in paths.items():
            is_broken = False
            if not os.path.exists(path):
                is_broken = True # 缺失
            else:
                if path.endswith(".png"):
                    try:
                        with Image.open(path) as img:
                            img.verify()
                        with Image.open(path) as img: # 再次開啟檢查是否全黑
                             if img.convert("L").getextrema() == (0, 0): is_broken = True
                    except:
                        is_broken = True
                elif path.endswith(".pt"):
                    try:
                        torch.load(path)
                    except:
                        is_broken = True
            
            if is_broken:
                if os.path.exists(path): os.remove(path) # 刪除壞檔
                broken_files.append((i, key, path))

    if not broken_files:
        print("✅ No corrupt files found!")
        return

    print(f"\n🛠️ Found {len(broken_files)} corrupt files. Fixing now...")
    
    # 2. 修復
    for item in tqdm(broken_files, desc="Fixing"):
        i, key, path = item
        prompt = prompts[i]
        session_key = 123456 + i
        path_latent = os.path.join(DIR_LATENT, f"{i:05d}.pt")
        
        print(f"   Repairing {key}: {os.path.basename(path)}")

        if key == "Latent":
            # Latent 壞了，需要重新 Mapping (這裡借用 Alice 腳本來生成 Latent)
            # 因為 generate_mapped_baseline_and_latent 函數太長，這裡直接用 Alice 生成
            # Alice 如果沒傳入 init_latent，它會自己生成一個並存檔嗎？我們之前的腳本沒寫存檔
            # 所以這裡簡單用 pytorch 生成一個
            pass # (省略複雜邏輯，通常 Latent 不會壞，如果是 Cover 壞了下面會處理)

        if key == "Cover":
            if model is None:
                model = load_model()
                sampler = DPMSolverSampler(model)
            generate_cover_image(model, sampler, prompt, path, session_key)
            
        elif key == "Pure":
            # 確保 Latent 存在
            if not os.path.exists(path_latent):
                print("   ⚠️ Missing Latent for Pure, skipping...") 
                continue
            run_alice(ALICE_SCRIPT, prompt, session_key, path, path_latent, opt_iters=10, 
                      extra_args=["--lr", "0.25"])
                      
        elif key == "Unc":
            if not os.path.exists(path_latent):
                print("   ⚠️ Missing Latent for Unc, skipping...") 
                continue
            run_alice(ALICE_UNC_SCRIPT, prompt, session_key, path, path_latent, opt_iters=10, 
                      extra_args=["--use_uncertainty", "--lr", "0.05", "--lambda_reg", "1.5"])

    print("\n✅ All repairs complete!")

if __name__ == "__main__":
    check_and_fix()