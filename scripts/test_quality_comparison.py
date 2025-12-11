import os
import sys
import torch
import numpy as np
import cv2
import lpips
import ssl
import shutil
import gc
from collections import defaultdict
from pytorch_fid import fid_score
from omegaconf import OmegaConf
from PIL import Image
from torch import autocast
from tqdm import tqdm
import json
import random


# === 1. 路徑設定 ===
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

# 導入模組
try:
    from ldm.util import instantiate_from_config
    from ldm.models.diffusion.dpm_solver import DPMSolverSampler
    from mapping_module import ours_mapping 
    # 【關鍵加速】直接導入 Alice 生成函數，不再使用 subprocess
    from pure_alice_uncertainty_fixed import generate_alice_image 
except ImportError as e:
    print(f"❌ 無法導入模組: {e}")
    sys.exit(1)

# BRISQUE
try:
    from piq import brisque
    BRISQUE_AVAILABLE = True
except ImportError:
    BRISQUE_AVAILABLE = False

ssl._create_default_https_context = ssl._create_unverified_context

# === 配置 ===
TOTAL_IMAGES = 50 

MAS_GRDH_PATH = PARENT_DIR
CKPT_PATH = "/home/vcpuser/netdrive/Workspace/stt/mas_GRDH/weights/v1-5-pruned.ckpt"
if not os.path.exists(CKPT_PATH):
    CKPT_PATH = os.path.join(MAS_GRDH_PATH, "weights/v1-5-pruned.ckpt")
CONFIG_PATH = os.path.join(MAS_GRDH_PATH, "configs/stable-diffusion/ldm.yaml")

DIR_REAL_COCO = os.path.join(CURRENT_DIR, "coco_val2017")
PATH_CAPTIONS = os.path.join(CURRENT_DIR, "coco_annotations", "captions_val2017.json")

OUTPUT_ROOT = os.path.join(MAS_GRDH_PATH, "outputs", "paper_repro_results")
DIR_COVER = os.path.join(OUTPUT_ROOT, "cover_sd")
DIR_MAPPED = os.path.join(OUTPUT_ROOT, "mapped_base")
DIR_PURE = os.path.join(OUTPUT_ROOT, "ours_pure")
DIR_UNC = os.path.join(OUTPUT_ROOT, "ours_unc")
DIR_REAL_RESIZED = os.path.join(OUTPUT_ROOT, "real_coco_resized")
DIR_TEMP = os.path.join(OUTPUT_ROOT, "temp")
DIR_LATENT = os.path.join(OUTPUT_ROOT, "latents") 

class QualityEvaluator:
    def __init__(self, device="cuda"):
        self.device = device
        print("⚙️  Loading LPIPS model...")
        self.loss_fn_alex = lpips.LPIPS(net='alex').to(device)

    def load_img_tensor(self, path, range_norm=True):
        try:
            img = cv2.imread(path)
            if img is None: return None
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (512, 512))
            img = img / 255.0
            if range_norm: img = img * 2.0 - 1.0
            img = np.transpose(img, (2, 0, 1))
            return torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(self.device)
        except: return None

    def calculate_lpips(self, path_ref, path_target):
        t_ref = self.load_img_tensor(path_ref, range_norm=True)
        t_tar = self.load_img_tensor(path_target, range_norm=True)
        if t_ref is None or t_tar is None: return None
        with torch.no_grad():
            return self.loss_fn_alex(t_ref, t_tar).item()

    def calculate_brisque(self, path_target):
        if not BRISQUE_AVAILABLE: return 0.0
        t_tar = self.load_img_tensor(path_target, range_norm=False)
        if t_tar is None: return 0.0
        try:
            with torch.no_grad():
                return brisque(t_tar, data_range=1.0, reduction='none').item()
        except: return 0.0

def load_model():
    print(f"⏳ Loading SD Model (ONCE for all)...")
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

def load_coco_prompts(json_path, limit=50):
    if not os.path.exists(json_path):
        print("❌ Captions missing.")
        sys.exit(1)
    with open(json_path, 'r') as f:
        data = json.load(f)
    captions = [item['caption'] for item in data['annotations']]
    random.shuffle(captions)
    return captions[:limit]

def generate_cover_image(model, sampler, prompt, out_path, seed):
    if os.path.exists(out_path): return
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

def generate_mapped_baseline_and_latent(model, sampler, prompt, session_key, payload_data, out_img_path, out_latent_path):
    if os.path.exists(out_img_path) and os.path.exists(out_latent_path): return

    device = torch.device("cuda")
    
    # Payload processing (Already passed as bytes)
    bits = np.unpackbits(np.frombuffer(payload_data, dtype=np.uint8))
    if len(bits) < 16384: bits = np.pad(bits, (0, 16384 - len(bits)), 'constant')
    bits = bits[:16384].reshape(1, 4, 64, 64)

    mapper = ours_mapping(bits=1)
    z_target_numpy = mapper.encode_secret(secret_message=bits, seed_kernel=session_key, seed_shuffle=session_key + 999)
    z_target = torch.from_numpy(z_target_numpy).float().to(device)
    
    torch.save(z_target, out_latent_path)

    c = model.get_learned_conditioning([prompt])
    uc = model.get_learned_conditioning([""])
    with torch.no_grad(), autocast("cuda"):
        z_0_final, _ = sampler.sample(steps=20, conditioning=c, batch_size=1, shape=(4, 64, 64),
                                      unconditional_guidance_scale=5.0, unconditional_conditioning=uc,
                                      x_T=z_target, DPMencode=False, DPMdecode=True, verbose=False)
        x_samples = model.decode_first_stage(z_0_final)
    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
    Image.fromarray((x_samples[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)).save(out_img_path)

def resize_images(src_dir, dst_dir, target_size=(512, 512), max_images=None):
    if not os.path.exists(src_dir): return
    os.makedirs(dst_dir, exist_ok=True)
    files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.png'))]
    if max_images: files = files[:max_images]
    if len(os.listdir(dst_dir)) >= len(files): return
    print(f"⚙️  Resizing {len(files)} images for FID...")
    for f in tqdm(files, desc="Resizing"):
        try:
            with Image.open(os.path.join(src_dir, f)) as img:
                img.convert('RGB').resize(target_size, Image.BICUBIC).save(os.path.join(dst_dir, f))
        except: pass

def main():
    print(f"🚀 High-Speed Paper Reproduction: {TOTAL_IMAGES} images 🚀")
    
    for d in [DIR_COVER, DIR_MAPPED, DIR_PURE, DIR_UNC, DIR_REAL_RESIZED, DIR_TEMP, DIR_LATENT]:
        os.makedirs(d, exist_ok=True)
        
    payload_path = os.path.join(DIR_TEMP, "payload.dat")
    if not os.path.exists(payload_path):
        with open(payload_path, "wb") as f: f.write(os.urandom(2048))
        
    # Pre-read payload for function calls
    with open(payload_path, "rb") as f: 
        raw_data = f.read()
        CAPACITY_BYTES = 16384 // 8 
        if len(raw_data) > CAPACITY_BYTES - 2: raw_data = raw_data[:CAPACITY_BYTES-2]
        length_header = len(raw_data).to_bytes(2, 'big')
        final_payload = length_header + raw_data
        if len(final_payload) < CAPACITY_BYTES: final_payload += b'\x00' * (CAPACITY_BYTES - len(final_payload))
        payload_data = final_payload

    prompts = load_coco_prompts(PATH_CAPTIONS, limit=TOTAL_IMAGES)

    # === 強制清除舊圖 ===
    print("\n🧹 Cleaning up old Alice results...")
    if os.path.exists(DIR_PURE): shutil.rmtree(DIR_PURE)
    if os.path.exists(DIR_UNC): shutil.rmtree(DIR_UNC)
    os.makedirs(DIR_PURE, exist_ok=True)
    os.makedirs(DIR_UNC, exist_ok=True)

    # === 單一進程生成所有圖片 (極速模式) ===
    print("\n📦 Loading Model (ONCE for all)...")
    model = load_model()
    sampler = DPMSolverSampler(model)

    print("\n📸 Generating Images (Batch Processing)...")
    
    for i, prompt in tqdm(enumerate(prompts), total=len(prompts)):
        session_key = 123456 + i
        
        path_cover = os.path.join(DIR_COVER, f"{i:05d}.png")
        path_mapped = os.path.join(DIR_MAPPED, f"{i:05d}.png")
        path_pure = os.path.join(DIR_PURE, f"{i:05d}.png")
        path_unc = os.path.join(DIR_UNC, f"{i:05d}.png")
        path_latent = os.path.join(DIR_LATENT, f"{i:05d}.pt")
        
        # 1. Baseline & Mapped
        if not os.path.exists(path_cover):
            generate_cover_image(model, sampler, prompt, path_cover, seed=session_key)
        
        if not os.path.exists(path_mapped):
            generate_mapped_baseline_and_latent(model, sampler, prompt, session_key, payload_data, path_mapped, path_latent)

        # 2. Pure (LR=0.25)
        # 直接呼叫函數，不啟動新進程
        if not os.path.exists(path_pure):
            generate_alice_image(
                model, sampler, prompt, session_key, payload_data, path_pure, 
                init_latent_path=path_latent,
                opt_iters=10, lr=0.25, lambda_reg=0.0, use_uncertainty=False
            )

        # 3. Unc (LR=0.05, Reg=1.5)
        if not os.path.exists(path_unc):
            generate_alice_image(
                model, sampler, prompt, session_key, payload_data, path_unc, 
                init_latent_path=path_latent,
                opt_iters=10, lr=0.05, lambda_reg=1.5, use_uncertainty=True
            )

    print("\n✅ Generation Complete! Starting Metrics Calculation...")
    
    # 釋放模型
    del model
    del sampler
    torch.cuda.empty_cache()
    gc.collect()

    # === Metrics ===
    print("\n[Phase 3] Calculating Metrics...")
    evaluator = QualityEvaluator()
    stats = defaultdict(lambda: {"Cover": [], "Pure": [], "Unc": []})

    for i in tqdm(range(len(prompts))):
        path_cover = os.path.join(DIR_COVER, f"{i:05d}.png")
        path_mapped = os.path.join(DIR_MAPPED, f"{i:05d}.png")
        path_pure = os.path.join(DIR_PURE, f"{i:05d}.png")
        path_unc = os.path.join(DIR_UNC, f"{i:05d}.png")

        stats["BRISQUE"]["Cover"].append(evaluator.calculate_brisque(path_cover))
        stats["BRISQUE"]["Pure"].append(evaluator.calculate_brisque(path_pure))
        stats["BRISQUE"]["Unc"].append(evaluator.calculate_brisque(path_unc))

        stats["LPIPS"]["Pure"].append(evaluator.calculate_lpips(path_mapped, path_pure))
        stats["LPIPS"]["Unc"].append(evaluator.calculate_lpips(path_mapped, path_unc))

    # FID
    print("   Resizing Real Images...")
    resize_images(DIR_REAL_COCO, DIR_REAL_RESIZED, max_images=TOTAL_IMAGES)
    
    print("   Calculating FID (num_workers=0)...")
    def calc_fid(p1, p2):
        try: return fid_score.calculate_fid_given_paths([p1, p2], batch_size=50, device="cuda", dims=2048, num_workers=0)
        except: return 999.99

    fid_cover = calc_fid(DIR_REAL_RESIZED, DIR_COVER)
    fid_pure = calc_fid(DIR_REAL_RESIZED, DIR_PURE)
    fid_unc = calc_fid(DIR_REAL_RESIZED, DIR_UNC)

    # === Report ===
    print("\n" + "="*80)
    print("FINAL RESULT: Paper Reproduction (Metric: COCO Prompts)")
    print("Optimization: Pure (LR=0.25) vs Ours (LR=0.05, Reg=1.5)")
    print("-" * 80)
    
    print(f"FID (↓) | Quality vs Real COCO")
    print(f"  Cover (SD) : {fid_cover:.4f}")
    print(f"  Ours (Pure): {fid_pure:.4f}")
    print(f"  Ours (Unc) : {fid_unc:.4f} ({'✅ Better' if fid_unc < fid_pure else '🔻 Worse'})")
    print("-" * 80)
    
    avg_brisque_cover = np.mean(stats["BRISQUE"]["Cover"])
    avg_brisque_pure = np.mean(stats["BRISQUE"]["Pure"])
    avg_brisque_unc = np.mean(stats["BRISQUE"]["Unc"])
    print(f"BRISQUE (↓) | Naturalness")
    print(f"  Cover (SD) : {avg_brisque_cover:.2f}")
    print(f"  Ours (Pure): {avg_brisque_pure:.2f}")
    print(f"  Ours (Unc) : {avg_brisque_unc:.2f} ({'✅ Better' if avg_brisque_unc < avg_brisque_pure else '🔻 Worse'})")
    print("-" * 80)

    avg_lpips_pure = np.mean(stats["LPIPS"]["Pure"])
    avg_lpips_unc = np.mean(stats["LPIPS"]["Unc"])
    print(f"LPIPS (↓) | Distortion vs Mapped Baseline")
    print(f"  Ours (Pure): {avg_lpips_pure:.4f}")
    print(f"  Ours (Unc) : {avg_lpips_unc:.4f} ({'✅ Better' if avg_lpips_unc < avg_lpips_pure else '🔻 Worse'})")
    print("="*80)

if __name__ == "__main__":
    main()