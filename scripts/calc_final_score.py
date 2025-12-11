import os
import sys
import torch
import numpy as np
import lpips
from collections import defaultdict
from pytorch_fid import fid_score
from PIL import Image
from tqdm import tqdm

# === 路徑設定 ===
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR)

# 嘗試導入 BRISQUE
try:
    from piq import brisque
    BRISQUE_AVAILABLE = True
except ImportError:
    BRISQUE_AVAILABLE = False
    print("⚠️ piq not found, skipping BRISQUE")

# 資料夾路徑 (必須與之前一致)
OUTPUT_ROOT = os.path.join(MAS_GRDH_PATH, "outputs", "paper_repro_results") if 'MAS_GRDH_PATH' in globals() else os.path.join(PARENT_DIR, "outputs", "paper_repro_results")
DIR_COVER = os.path.join(OUTPUT_ROOT, "cover_sd")
DIR_MAPPED = os.path.join(OUTPUT_ROOT, "mapped_base")
DIR_PURE = os.path.join(OUTPUT_ROOT, "ours_pure")
DIR_UNC = os.path.join(OUTPUT_ROOT, "ours_unc")
DIR_REAL_COCO = os.path.join(CURRENT_DIR, "coco_val2017")
DIR_REAL_RESIZED = os.path.join(OUTPUT_ROOT, "real_coco_resized")

class QualityEvaluator:
    def __init__(self, device="cuda"):
        self.device = device
        print("⚙️  Loading LPIPS model...")
        self.loss_fn_alex = lpips.LPIPS(net='alex').to(device)

    def load_img_tensor(self, path, range_norm=True):
        try:
            # 使用 PIL 讀取比較穩定
            img = Image.open(path).convert("RGB")
            img = np.array(img).astype(np.float32) / 255.0
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

def resize_images(src_dir, dst_dir, target_size=(512, 512), max_images=1000):
    if not os.path.exists(src_dir): 
        print(f"❌ Real images not found at {src_dir}")
        return
    os.makedirs(dst_dir, exist_ok=True)
    
    files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.png'))]
    if max_images: files = files[:max_images]
    
    if len(os.listdir(dst_dir)) >= len(files): 
        print("   ✅ Resized Real Images ready.")
        return

    print(f"⚙️  Resizing {len(files)} real images for FID...")
    for f in tqdm(files):
        try:
            with Image.open(os.path.join(src_dir, f)) as img:
                img.convert('RGB').resize(target_size, Image.BICUBIC).save(os.path.join(dst_dir, f))
        except: pass

def main():
    print("🚀 Calculating Final Scores (Only Calculation)...")
    evaluator = QualityEvaluator()
    
    # 確保每個資料夾都有 1000 張
    for d in [DIR_COVER, DIR_PURE, DIR_UNC]:
        count = len([f for f in os.listdir(d) if f.endswith(".png")])
        print(f"   Checking {os.path.basename(d)}: {count} images")
        if count < 1000:
            print(f"   ⚠️ Warning: Only {count} images found. FID might be inaccurate.")

    # 1. BRISQUE & LPIPS
    print("\n📊 Calculating Perceptual Metrics (BRISQUE & LPIPS)...")
    stats = defaultdict(lambda: {"Cover": [], "Pure": [], "Unc": []})
    
    # 讀取檔案列表 (假設檔名一致)
    files = sorted([f for f in os.listdir(DIR_COVER) if f.endswith(".png")])
    files = files[:1000] # 確保只算 1000 張

    for f in tqdm(files):
        p_cover = os.path.join(DIR_COVER, f)
        p_mapped = os.path.join(DIR_MAPPED, f)
        p_pure = os.path.join(DIR_PURE, f)
        p_unc = os.path.join(DIR_UNC, f)

        # BRISQUE
        stats["BRISQUE"]["Cover"].append(evaluator.calculate_brisque(p_cover))
        stats["BRISQUE"]["Pure"].append(evaluator.calculate_brisque(p_pure))
        stats["BRISQUE"]["Unc"].append(evaluator.calculate_brisque(p_unc))

        # LPIPS
        stats["LPIPS"]["Pure"].append(evaluator.calculate_lpips(p_mapped, p_pure))
        stats["LPIPS"]["Unc"].append(evaluator.calculate_lpips(path_ref=p_mapped, path_target=p_unc))

    # 2. FID
    print("\n📊 Calculating FID (This is the heavy part)...")
    resize_images(DIR_REAL_COCO, DIR_REAL_RESIZED, max_images=1000)
    
    def calc_fid(path_fake, label):
        print(f"   FID: {label}...")
        try:
            # num_workers=0 絕對穩
            return fid_score.calculate_fid_given_paths([DIR_REAL_RESIZED, path_fake], batch_size=50, device="cuda", dims=2048, num_workers=0)
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return 999.99

    fid_cover = calc_fid(DIR_COVER, "Cover (Baseline)")
    fid_pure = calc_fid(DIR_PURE, "Ours (Pure)")
    fid_unc = calc_fid(DIR_UNC, "Ours (Uncertainty)")

    # === Final Report ===
    print("\n" + "="*80)
    print("🏆 FINAL THESIS TABLE DATA 🏆")
    print("-" * 80)
    
    # FID
    print(f"FID (↓) | Quality vs Real COCO")
    print(f"  Baseline (SD): {fid_cover:.4f}")
    print(f"  Ours (Pure)  : {fid_pure:.4f}")
    print(f"  Ours (Unc)   : {fid_unc:.4f}")
    print("-" * 80)
    
    # BRISQUE
    avg_brisque_cover = np.mean(stats["BRISQUE"]["Cover"])
    avg_brisque_pure = np.mean(stats["BRISQUE"]["Pure"])
    avg_brisque_unc = np.mean(stats["BRISQUE"]["Unc"])
    print(f"BRISQUE (↓) | Naturalness")
    print(f"  Baseline (SD): {avg_brisque_cover:.2f}")
    print(f"  Ours (Pure)  : {avg_brisque_pure:.2f}")
    print(f"  Ours (Unc)   : {avg_brisque_unc:.2f}")
    print("-" * 80)

    # LPIPS
    avg_lpips_pure = np.mean(stats["LPIPS"]["Pure"])
    avg_lpips_unc = np.mean(stats["LPIPS"]["Unc"])
    print(f"LPIPS (↓) | Distortion Cost")
    print(f"  Ours (Pure)  : {avg_lpips_pure:.4f}")
    print(f"  Ours (Unc)   : {avg_lpips_unc:.4f}")
    print("="*80)

if __name__ == "__main__":
    main()