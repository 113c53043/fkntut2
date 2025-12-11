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
from scipy.linalg import qr
from scipy.stats import norm

# === 1. 內嵌 Mapping 類別 (確保測試腳本也能獨立運行) ===
class mapping_module:
    def __init__(self, need_uniform_sampler=False, need_gaussian_sampler=False, bits=1, seed=None):
        self.need_uniform_sampler = need_uniform_sampler
        self.need_gaussian_sampler = need_gaussian_sampler
        self.bits = bits
        self.bits_l = 2 ** bits
        self.seed = seed

class ours_mapping_tuned(mapping_module):
    def __init__(self, bits=1, scale=1.0, clip_range=3.5):
        super(ours_mapping_tuned, self).__init__(bits=bits)
        self.bits_mean = (self.bits_l - 1) / 2
        self.bits_std = ((self.bits_l ** 2 - 1) / 12) ** 0.5
        self.scale = scale          
        self.clip_range = clip_range 

    def _get_random_kernel(self, seed_kernel, kernel_shape):
        ori_seed = np.random.get_state()[1][0]
        np.random.seed(seed_kernel)
        H = np.random.randn(*kernel_shape)
        Q, r = qr(H)
        kernel = Q
        np.random.seed(ori_seed)
        return kernel

    def _random_shuffle(self, ori_input, seed_shuffle, reverse=False):
        ori_seed = np.random.get_state()[1][0]
        np.random.seed(seed_shuffle)
        ori_shape = ori_input.shape
        ori_input = ori_input.flatten()
        ori_order = np.arange(0, len(ori_input))
        shuffle_order = ori_order.copy()
        np.random.shuffle(shuffle_order)
        if reverse:
            sorted_shuffle_order = np.argsort(shuffle_order)
            reverse_order = ori_order[sorted_shuffle_order]
            out = ori_input[reverse_order]
        else:
            out = ori_input[shuffle_order]
        out = out.reshape(*ori_shape)
        np.random.seed(ori_seed)
        return out

    def encode_secret(self, secret_message, ori_sample=None, seed_kernel=None, seed_shuffle=None):
        kernel = self._get_random_kernel(seed_kernel=seed_kernel, kernel_shape=secret_message.shape[-2:])
        secret_re = (secret_message - self.bits_mean) / self.bits_std
        out = np.matmul(np.matmul(kernel, secret_re), kernel.transpose(-1, -2))
        out = self._random_shuffle(out, seed_shuffle=seed_shuffle)
        out = out * self.scale
        if self.clip_range is not None:
            out = np.clip(out, -self.clip_range, self.clip_range)
        return out

# === 2. 配置與導入 ===
try:
    from pytorch_fid import fid_score
    from piq import brisque
    PIQ_AVAILABLE = True
except ImportError:
    PIQ_AVAILABLE = False

CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

try:
    from ldm.util import instantiate_from_config
    from ldm.models.diffusion.dpm_solver import DPMSolverSampler
    from pure_alice_dynamic import generate_alice_image
    from robust_eval import identity, jpeg, resize, gblur, awgn
    from utils import load_512
except ImportError as e:
    print(f"❌ 導入失敗: {e}")
    sys.exit(1)

MAS_GRDH_PATH = PARENT_DIR
CKPT_PATH = "/home/vcpuser/netdrive/Workspace/stt/mas_GRDH/weights/v1-5-pruned.ckpt" 
if not os.path.exists(CKPT_PATH):
    CKPT_PATH = os.path.join(MAS_GRDH_PATH, "weights/v1-5-pruned.ckpt")
CONFIG_PATH = os.path.join(MAS_GRDH_PATH, "configs/stable-diffusion/ldm.yaml")
DIR_REAL_COCO = os.path.join(CURRENT_DIR, "coco_val2017")
PATH_CAPTIONS = os.path.join(CURRENT_DIR, "coco_annotations", "captions_val2017.json")

OUTPUT_ROOT = os.path.join(MAS_GRDH_PATH, "outputs", "all_in_one_v4_debug")
DIR_COVER = os.path.join(OUTPUT_ROOT, "cover")
DIR_BASELINE = os.path.join(OUTPUT_ROOT, "baseline")
DIR_OURS = os.path.join(OUTPUT_ROOT, "ours")
DIR_REAL_RESIZED = os.path.join(OUTPUT_ROOT, "real_resized")
DIR_ATTACKED = os.path.join(OUTPUT_ROOT, "attacked_temp")

TEST_SAMPLES = 50

def load_model():
    print(f"⏳ Loading SD Model...")
    config = OmegaConf.load(CONFIG_PATH)
    def recursive_fix(conf):
        if isinstance(conf, (dict, OmegaConf)):
            for key in conf.keys():
                if key == "image_size" and conf[key] == 32: conf[key] = 64
                recursive_fix(conf[key])
    recursive_fix(config.model)
    try:
        pl_sd = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    except TypeError:
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

def generate_baseline(model, sampler, prompt, key, payload, out_path):
    """
    Baseline (Txt2Img): 標準生成 + 隱寫嵌入 (無動態優化)
    """
    if os.path.exists(out_path): return
    torch.manual_seed(key)
    torch.cuda.manual_seed(key)
    
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))
    if len(bits) < 16384: bits = np.pad(bits, (0, 16384 - len(bits)), 'constant')
    bits = bits[:16384].reshape(1, 4, 64, 64)
    
    # Baseline 使用原始參數: scale=1.0, clip=None
    mapper = ours_mapping_tuned(bits=1, scale=1.0, clip_range=None) 
    z_numpy = mapper.encode_secret(secret_message=bits, seed_kernel=key, seed_shuffle=key + 999)
    z_target = torch.from_numpy(z_numpy).float().cuda()
    
    c = model.get_learned_conditioning([prompt])
    uc = model.get_learned_conditioning([""])
    
    with torch.no_grad(), autocast("cuda"):
        # 標準 DPM 生成，無迭代優化
        z_enc, _ = sampler.sample(steps=20, conditioning=c, batch_size=1, shape=(4,64,64),
                                  unconditional_guidance_scale=5.0, unconditional_conditioning=uc,
                                  x_T=z_target, DPMencode=False, DPMdecode=True, verbose=False)
        x_samples = model.decode_first_stage(z_enc)
    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
    Image.fromarray((x_samples[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)).save(out_path)

def load_prompts(limit):
    if not os.path.exists(PATH_CAPTIONS):
        print("❌ Captions file not found. Using dummy prompts.")
        return ["A photo of a cat"] * limit
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

def run_bob_decode(img_path, prompt, key, gt_path, is_baseline=False):
    scale_arg = "1.0" if is_baseline else "0.9"
    clip_arg = "0" if is_baseline else "3.0"
    
    # 關鍵：指向 pure_bob.py 的絕對路徑
    bob_script = os.path.join(MAS_GRDH_PATH, "pure_bob.py")
    if not os.path.exists(bob_script):
        print(f"\n❌ Error: pure_bob.py not found at {bob_script}")
        return 0.0

    cmd = [
        sys.executable, bob_script,
        "--img_path", img_path,
        "--prompt", prompt,
        "--secret_key", str(key),
        "--gt_path", gt_path,
        "--ckpt", CKPT_PATH,
        "--config", CONFIG_PATH,
        "--dpm_steps", "20",
        "--mapping_scale", scale_arg,
        "--mapping_clip", clip_arg
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        match = re.search(r"Bit Accuracy.*?: (\d+\.\d+)%", res.stdout)
        if match:
            return float(match.group(1))
        else:
            # 如果解析失敗，打印錯誤訊息以便 Debug
            if res.returncode != 0:
                print(f"\n[Bob Failed] {res.stderr[:200]}...") 
            return 0.0
    except Exception as e:
        print(f"\n[Subprocess Error] {e}")
        return 0.0

def main():
    print(f"🚀 Rigorous Test V4: {TEST_SAMPLES} Samples 🚀")
    print(f"Comparing: Baseline (Txt2Img) vs Ours (Dynamic)")
    
    for d in [DIR_COVER, DIR_BASELINE, DIR_OURS, DIR_REAL_RESIZED, DIR_ATTACKED]:
        os.makedirs(d, exist_ok=True)
        
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
    
    records = []
    
    for i, p in enumerate(tqdm(prompts)):
        key = 10000 + i
        p_cover = os.path.join(DIR_COVER, f"{i:05d}.png")
        p_base = os.path.join(DIR_BASELINE, f"{i:05d}.png")
        p_ours = os.path.join(DIR_OURS, f"{i:05d}.png")
        gt_bits_path = p_ours + ".gt_bits.npy"
        np.save(gt_bits_path, np.frombuffer(final_payload, dtype=np.uint8))
        
        generate_cover(model, sampler, p, p_cover, key)
        generate_baseline(model, sampler, p, key, final_payload, p_base)
        if not os.path.exists(p_ours):
            generate_alice_image(model, sampler, p, key, final_payload, p_ours, opt_iters=15, use_uncertainty=True)
            
        records.append({"id": i, "prompt": p, "key": key, "cover": p_cover, "base": p_base, "ours": p_ours, "gt": gt_bits_path})
        
    del model, sampler
    torch.cuda.empty_cache()
    gc.collect()
    
    # 2. 畫質評估
    print("\n[Phase 2] Evaluating Quality...")
    metrics = defaultdict(lambda: defaultdict(list))
    
    for r in tqdm(records):
        metrics["BRISQUE"]["Cover"].append(get_brisque(r["cover"]))
        metrics["BRISQUE"]["Baseline"].append(get_brisque(r["base"]))
        metrics["BRISQUE"]["Ours"].append(get_brisque(r["ours"]))
        
        img_c = load_512(r["cover"]).cuda()
        img_b = load_512(r["base"]).cuda()
        img_o = load_512(r["ours"]).cuda()
        with torch.no_grad():
            metrics["LPIPS"]["Baseline"].append(lpips_fn(img_c, img_b).item())
            metrics["LPIPS"]["Ours"].append(lpips_fn(img_c, img_o).item())
            
    # FID
    resize_real_images(DIR_REAL_COCO, DIR_REAL_RESIZED, TEST_SAMPLES)
    try:
        fid_base = fid_score.calculate_fid_given_paths([DIR_REAL_RESIZED, DIR_BASELINE], 50, "cuda", 2048, 0)
        fid_ours = fid_score.calculate_fid_given_paths([DIR_REAL_RESIZED, DIR_OURS], 50, "cuda", 2048, 0)
    except: fid_base, fid_ours = 999.0, 999.0
    
    # 3. 魯棒性評估
    print("\n[Phase 3] Evaluating Robustness...")
    attacks = [
        ("Identity", identity, None),
        ("JPEG(80)", jpeg, 80),
        ("Resize(0.5)", resize, 0.5),
        ("GBlur(3x3)", gblur, 3),
        ("AWGN(0.01)", awgn, 0.01)
    ]
    
    robust = defaultdict(lambda: defaultdict(list))
    
    for name, attack_fn, param in attacks:
        print(f"  Testing {name}...")
        for r in tqdm(records):
            tmp_name = os.path.join(DIR_ATTACKED, f"temp_{r['id']}")
            
            # Baseline (is_baseline=True)
            img_b = load_512(r["base"]).cuda()
            attack_fn(img_b, param, tmp_image_name=tmp_name)
            att_file = tmp_name + (".jpg" if "JPEG" in name else ".png")
            robust[name]["Baseline"].append(run_bob_decode(att_file, r["prompt"], r["key"], r["gt"], is_baseline=True))
            
            # Ours (is_baseline=False)
            img_o = load_512(r["ours"]).cuda()
            attack_fn(img_o, param, tmp_image_name=tmp_name)
            att_file = tmp_name + (".jpg" if "JPEG" in name else ".png")
            robust[name]["Ours"].append(run_bob_decode(att_file, r["prompt"], r["key"], r["gt"], is_baseline=False))

    # === 最終報告 ===
    print("\n" + "="*80)
    print(f"📊 FINAL REPORT (Samples: {TEST_SAMPLES})")
    print("="*80)
    print(f"{'Metric':<15} | {'Baseline (Txt2Img)':<20} | {'Ours (Dynamic)':<20}")
    print("-" * 80)
    print(f"{'FID (vs COCO)':<15} | {fid_base:<20.4f} | {fid_ours:<20.4f}")
    print(f"{'BRISQUE':<15} | {np.mean(metrics['BRISQUE']['Baseline']):<20.4f} | {np.mean(metrics['BRISQUE']['Ours']):<20.4f}")
    print(f"{'LPIPS':<15} | {np.mean(metrics['LPIPS']['Baseline']):<20.4f} | {np.mean(metrics['LPIPS']['Ours']):<20.4f}")
    print("-" * 80)
    print("Robustness (Bit Accuracy):")
    for name, _, _ in attacks:
        acc_b = np.mean(robust[name]["Baseline"])
        acc_o = np.mean(robust[name]["Ours"])
        print(f"  {name:<13} | {acc_b:<20.2f}% | {acc_o:<20.2f}%")
    print("="*80)

if __name__ == "__main__":
    main()