import os
import sys
import numpy as np
import shutil 
import torch
from collections import defaultdict
import gc
import cv2
import lpips
from tqdm import tqdm
from PIL import Image
from torch import autocast
import re

# === 1. 路徑設定 (最優先執行) ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR) 
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

# 引用模組
try:
    from pytorch_fid import fid_score
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config
    from ldm.models.diffusion.dpm_solver import DPMSolverSampler
    from mapping_module import ours_mapping # Bob 解碼需要這個
except ImportError as e:
    print(f"⚠️ Import Warning: {e}")

try:
    from piq import brisque
    BRISQUE_AVAILABLE = True
except ImportError:
    BRISQUE_AVAILABLE = False

try:
    from robust_eval import identity, storage, resize, jpeg, mblur, gblur, awgn, crop, rotation
    from utils import load_512
except ImportError:
    pass

# === 2. 配置 ===
MAS_GRDH_PATH = PARENT_DIR 
CKPT_PATH = "/home/vcpuser/netdrive/Workspace/stt/mas_GRDH/weights/v1-5-pruned.ckpt"
if not os.path.exists(CKPT_PATH):
    CKPT_PATH = os.path.join(MAS_GRDH_PATH, "weights/v1-5-pruned.ckpt")
CONFIG_PATH = os.path.join(MAS_GRDH_PATH, "configs/stable-diffusion/ldm.yaml")
PROMPT_FILE_LIST = os.path.join(MAS_GRDH_PATH, "text_prompt_dataset", "coco_dataset.txt")
ALICE_SCRIPT = os.path.join(MAS_GRDH_PATH, "pure_alice_final.py")

# [重要] 請確認您的舊圖片是否在這個資料夾，如果不是，請手動搬移過來
OUTPUT_DIR = os.path.join(MAS_GRDH_PATH, "outputs", "systematic_study_final_opt")
DIR_REAL_COCO = os.path.join(MAS_GRDH_PATH, "scripts", "coco_val2017") 

# === 關鍵設置 ===
# Phase 1: 生成 2000 張 (FID用)
TOTAL_GENERATE = 2000 
# Phase 2: 評估 500 張 (Acc用)
TOTAL_EVALUATE = 500

MODES = ["baseline", "pure", "fixed", "adaptive"]
SKIP_GENERATION_IF_EXISTS = True # 設為 True 開啟斷點續傳
RUN_ATTACK_AND_BOB = True
CALC_FID = True

ATTACK_SUITE = [
    (identity, [None], "Identity", ".png"),
    (jpeg, [50], "JPEG(50)", ".jpg"),
    (resize, [0.5], "Resize(0.5)", ".png"),
    (mblur, [5], "MBlur(5)", ".png"),
    (gblur, [5], "GBlur(5)", ".png"),
    (awgn, [0.05], "Noise(0.05)", ".png"),
    (crop, [0.2], "Crop(0.2)", ".png"),
    (rotation, [25], "Rot(25)", ".png"),
]

# 統一的 Negative Prompt
LONG_NEGATIVE_PROMPT = "worst quality, low quality, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, normal quality, jpeg artifacts, signature, watermark, username, blurry, bad feet, extra arms, extra legs, extra body, poorly drawn hands, missing arms, missing legs, extra hands, mangled fingers, extra fingers, disconnected limbs, mutated hands, long neck, duplicate, bad composition, malformed limbs, deformed, mutated, ugly, disgusting, amputation, cartoon, anime, 3d, illustration, talking, two bodies, double torso, three arms, three legs, bad framing, mutated face, deformed face, cross-eyed, body out of frame, cloned face, disfigured, fused fingers, too many fingers, long fingers, gross proportions, poorly drawn face, text focus, bad focus, out of focus, extra nipples, missing nipples, fused nipples, extra breasts, enlarged breasts, deformed breasts, bad shadow, overexposed, underexposed, bad lighting, color distortion, weird colors, dull colors, bad eyes, dead eyes, asymmetrical eyes, hollow eyes, collapsed eyes, mutated eyes, distorted iris, wrong eye position, wrong teeth, crooked teeth, melted teeth, distorted mouth, wrong lips, mutated lips, broken lips, twisted mouth, bad hair, coarse hair, messy hair, artifact hair, unnatural hair texture, missing hair, polygon hair, bad skin, oily skin, plastic skin, uneven skin, dirty skin, pores, face holes, oversharpen, overprocessed, nsfw, extra tongue, long tongue, split tongue, bad tongue, distorted tongue, blurry background, messy background, multiple heads, split head, fused head, broken head, missing head, duplicated head, wrong head, loli, child, kid, underage, boy, girl, infant, toddler, baby, baby face, young child, teen, 3D render, extra limb, twisted limb, broken limb, warped limb, oversized limb, undersized limb, smudge, glitch, errors, canvas frame, cropped head, cropped face, cropped body, depth-of-field error, weird depth, lens distortion, chromatic aberration, duplicate face, wrong face, face mismatch, hands behind back, incorrect fingers, extra joint, broken joint, doll-like, mannequin, porcelain skin, waxy skin, clay texture, incorrect grip, wrong pose, unnatural pose, floating object, floating limbs, floating head, missing shadow, unnatural shadow, dislocated shoulder, bad cloth, cloth error, clothing glitch, unnatural clothing folds, stretched fabric, corrupted texture, mosaic, censored, body distortion, bent spine, malformed spine, unnatural spine angle, twisted waist, extra waist, glowing eyes, horror eyes, scary face, mutilated, blood, gore, wounds, injury, amputee, long body, short body, bad perspective, impossible perspective, broken perspective, wrong angle, disfigured eyes, lazy eye, cyclops, extra eye, mutated body, malformed body, clay skin, huge head, tiny head, uneven head, incorrect anatomy, missing torso, half torso, torso distortion"

# === 3. 核心優化：常駐模型載入 ===
def load_shared_model():
    print(f"⏳ Loading Shared SD Model (Once for all)...")
    config = OmegaConf.load(CONFIG_PATH)
    
    # 修正 Config 中的 image_size
    def recursive_fix(conf):
        if isinstance(conf, (dict, OmegaConf)):
            for key in conf.keys():
                if key == "image_size" and conf[key] == 32: conf[key] = 64
                recursive_fix(conf[key])
    recursive_fix(config.model)

    try:
        pl_sd = torch.load(CKPT_PATH, map_location="cpu")
    except:
        pl_sd = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
        
    sd = pl_sd["state_dict"] if "state_dict" in pl_sd else pl_sd
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model

# === 4. Bob 記憶體內解碼 (In-Memory Decoding) ===
def fast_bob_decode(model, sampler, img_tensor, prompt, secret_key, gt_bits_path):
    """直接使用已載入的模型進行解碼，不需重啟進程。"""
    try:
        c = model.get_learned_conditioning([prompt])
        uc = model.get_learned_conditioning([LONG_NEGATIVE_PROMPT])
        
        with torch.no_grad(), autocast("cuda"):
            init_latent = model.get_first_stage_encoding(model.encode_first_stage(img_tensor))
            z_rec, _ = sampler.sample(steps=20, conditioning=c, batch_size=1, shape=init_latent.shape[1:],
                                      unconditional_guidance_scale=5.0, unconditional_conditioning=uc,
                                      x_T=init_latent, DPMencode=True, DPMdecode=False, verbose=False)

        mapper = ours_mapping(bits=1)
        z_rec_numpy = z_rec.cpu().numpy()
        
        decoded_float = mapper.decode_secret_soft(
            z_rec_numpy, seed_kernel=secret_key, seed_shuffle=secret_key + 999
        )
        bits = np.round(decoded_float).astype(np.uint8).flatten()
        extracted_bytes = np.packbits(bits).tobytes()

        if not os.path.exists(gt_bits_path): return 0.0
        gt_bytes = np.load(gt_bits_path).tobytes()
        
        arr_a = np.unpackbits(np.frombuffer(extracted_bytes, dtype=np.uint8))
        arr_b = np.unpackbits(np.frombuffer(gt_bytes, dtype=np.uint8))
        
        min_len = min(len(arr_a), len(arr_b))
        matches = np.sum(arr_a[:min_len] == arr_b[:min_len])
        total_bits = max(len(arr_a), len(arr_b))
        
        return (matches / total_bits) * 100.0

    except Exception as e:
        return 0.0

# === 5. Alice 呼叫 (使用 subprocess 獨立運行) ===
import subprocess
def run_alice(prompt, session_key, out_path, payload_path, mode):
    # 若檔案存在且設定為跳過，則直接返回
    if os.path.exists(out_path) and SKIP_GENERATION_IF_EXISTS: return True
    
    # === [關鍵修正] 使用 Balanced SOTA 參數 ===
    if mode == "pure": lr, reg = "0.25", "0.0"
    elif mode == "fixed": lr, reg = "0.05", "1.5"
    elif mode == "adaptive": lr, reg = "0.12", "1.25" # Balanced SOTA
    else: lr, reg = "0.05", "1.5"

    cmd = [
        sys.executable, ALICE_SCRIPT,
        "--prompt", prompt, "--secret_key", str(session_key),
        "--payload_path", payload_path, "--outpath", out_path,
        "--ckpt", CKPT_PATH, "--config", CONFIG_PATH,
        "--opt_iters", "10", "--dpm_steps", "20",
        "--lr", lr, "--lambda_reg", reg, "--mode", mode
    ]
    try:
        env = os.environ.copy()
        python_path = env.get("PYTHONPATH", "")
        paths = [PARENT_DIR, os.path.join(PARENT_DIR, "scripts")]
        for p in paths:
            if p not in python_path: python_path = f"{p}{os.pathsep}{python_path}"
        env["PYTHONPATH"] = python_path
        
        subprocess.run(cmd, check=True, cwd=MAS_GRDH_PATH, capture_output=True, text=True, timeout=600, env=env)
        return True
    except: return False

# === 6. 其他輔助函式 ===
def generate_cover_image(model, sampler, prompt, out_path, seed):
    if os.path.exists(out_path) and SKIP_GENERATION_IF_EXISTS: return
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

def create_gt_bits_file(payload_path, out_gt_path):
    CAPACITY_BYTES = 16384 // 8
    with open(payload_path, "rb") as f: raw_data = f.read()
    payload_data = raw_data
    if len(payload_data) > CAPACITY_BYTES - 2: payload_data = payload_data[:CAPACITY_BYTES-2]
    length_header = len(payload_data).to_bytes(2, 'big')
    final_payload = length_header + payload_data
    if len(final_payload) < CAPACITY_BYTES: final_payload += b'\x00' * (CAPACITY_BYTES - len(final_payload))
    np.save(out_gt_path, np.frombuffer(final_payload, dtype=np.uint8))

class QualityEvaluator:
    def __init__(self):
        self.lpips_fn = lpips.LPIPS(net='alex').cuda()
    def calc_lpips(self, p1, p2):
        try:
            t1 = self._load(p1); t2 = self._load(p2)
            with torch.no_grad(): return self.lpips_fn(t1, t2).item()
        except: return 0.0
    def calc_brisque(self, p1):
        try:
            t1 = self._load(p1, norm=False)
            with torch.no_grad(): return brisque(t1, data_range=1.0).item()
        except: return 0.0
    def _load(self, p, norm=True):
        img = cv2.imread(p); img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (512,512)) / 255.0
        if norm: img = img * 2 - 1
        return torch.tensor(img.transpose(2,0,1)).float().cuda().unsqueeze(0)

def resize_real_images(src_dir, dst_dir, target_size=(512, 512), max_images=None):
    if not os.path.exists(src_dir): return
    os.makedirs(dst_dir, exist_ok=True)
    files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.png'))]
    if max_images: files = files[:max_images]
    if len(os.listdir(dst_dir)) >= len(files): return
    print("⚙️ Resizing Real Images for FID...")
    for f in tqdm(files):
        try:
            with Image.open(os.path.join(src_dir, f)) as img:
                img.convert('RGB').resize(target_size, Image.BICUBIC).save(os.path.join(dst_dir, f))
        except: pass

# === Main ===
def main():
    print(f"🚀 FAST SYSTEMATIC TEST (In-Memory Bob) 🚀")
    print(f"Adaptive Params: LR=0.12, Reg=1.25 (Balanced Optimal)")
    print(f"Config: Gen={TOTAL_GENERATE}, Eval={TOTAL_EVALUATE}")
    print(f"Output Dir: {OUTPUT_DIR}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    subdirs = {m: os.path.join(OUTPUT_DIR, m) for m in MODES}
    subdirs["cover"] = os.path.join(OUTPUT_DIR, "cover") 
    dir_real_resized = os.path.join(OUTPUT_DIR, "real_coco_resized")
    for d in subdirs.values(): os.makedirs(d, exist_ok=True)
    
    prompts = []
    if os.path.exists(PROMPT_FILE_LIST):
        with open(PROMPT_FILE_LIST) as f: lines = [l.strip() for l in f if l.strip()]
        while len(prompts) < TOTAL_GENERATE: prompts.extend(lines)
    prompts = prompts[:TOTAL_GENERATE] if prompts else ["A futuristic city"] * TOTAL_GENERATE

    resize_real_images(DIR_REAL_COCO, dir_real_resized, max_images=TOTAL_GENERATE)

    # 1. 載入模型 (一次性)
    model = load_shared_model()
    sampler = DPMSolverSampler(model)
    evaluator = QualityEvaluator()

    # 2. 處理流程
    acc_results = defaultdict(lambda: defaultdict(list))
    qual_results = defaultdict(list)

    # Phase 1: Generation (Alice)
    # 這裡會自動跳過已存在的圖片 (SKIP_GENERATION_IF_EXISTS=True)
    print("\n--- Phase 1: Generating Images (Skipping existing) ---")
    
    # 檢查是否已有檔案，印出提示
    existing_count = len([f for f in os.listdir(subdirs['adaptive']) if f.endswith('.png')])
    if existing_count > 0:
        print(f"ℹ️ Found {existing_count} existing images in '{subdirs['adaptive']}'. Resume mode active.")
    else:
        print(f"ℹ️ No existing images found in target dir. Starting from scratch.")

    for i in tqdm(range(TOTAL_GENERATE)):
        session_key = 123456 + i
        payload_path = os.path.join(OUTPUT_DIR, f"p{i}.dat")
        if not os.path.exists(payload_path): 
            with open(payload_path, "wb") as f: f.write(os.urandom(2048))

        # Cover (使用 In-Memory 生成)
        cover_path = os.path.join(subdirs["cover"], f"{i:05d}.png")
        generate_cover_image(model, sampler, prompts[i], cover_path, seed=session_key)
        
        # Stego (使用 Subprocess 生成)
        for mode in MODES:
            out_p = os.path.join(subdirs[mode], f"{i:05d}.png")
            gt_p = out_p + ".gt_bits.npy"
            
            # 若檔案不存在才呼叫 Alice，並印出 Debug 訊息 (僅前幾次)
            if not os.path.exists(out_p):
                if i < 3: print(f"Generating missing: {out_p}")
                run_alice(prompts[i], session_key, out_p, payload_path, mode)
                
            create_gt_bits_file(payload_path, gt_p)

    # Phase 2: Evaluation (Bob)
    print("\n--- Phase 2: High-Speed Evaluation (In-Memory) ---")
    for i in tqdm(range(TOTAL_EVALUATE)):
        prompt = prompts[i]
        session_key = 123456 + i
        cover_path = os.path.join(subdirs["cover"], f"{i:05d}.png")

        for mode in MODES:
            out_p = os.path.join(subdirs[mode], f"{i:05d}.png")
            gt_p = out_p + ".gt_bits.npy"
            if not os.path.exists(out_p): continue
            
            # Quality
            if os.path.exists(cover_path):
                lpips_val = evaluator.calc_lpips(cover_path, out_p)
                brisque_val = evaluator.calc_brisque(out_p)
                qual_results[mode].append((lpips_val, brisque_val))

            # Robustness
            if RUN_ATTACK_AND_BOB:
                img_tensor = load_512(out_p).cuda()
                for atk_fn, args, atk_name, ext in ATTACK_SUITE:
                    # Attack
                    att_tensor = atk_fn(img_tensor.clone(), args[0])
                    # Decode (Fast)
                    acc = fast_bob_decode(model, sampler, att_tensor, prompt, session_key, gt_p)
                    acc_results[mode][atk_name].append(acc)
                del img_tensor

    # 3. FID
    print("\n📊 Calculating FID...")
    fid_scores = {}
    if CALC_FID and os.path.exists(dir_real_resized):
        del model 
        torch.cuda.empty_cache()
        for mode in MODES:
            try:
                fid = fid_score.calculate_fid_given_paths(
                    [dir_real_resized, subdirs[mode]], 
                    batch_size=50, device="cuda", dims=2048, num_workers=0
                )
                fid_scores[mode] = fid
            except: fid_scores[mode] = 0.0
    else:
        for mode in MODES: fid_scores[mode] = 0.0

    # 4. Report
    print("\n" + "="*120)
    print(f"FINAL REPORT (Gen N={TOTAL_GENERATE} | Eval N={TOTAL_EVALUATE})")
    print("="*120)
    
    headers = ["Metric"] + [m.upper() for m in MODES]
    print(f"{headers[0]:<20} | {headers[1]:<12} | {headers[2]:<12} | {headers[3]:<12} | {headers[4]:<12}")
    print("-" * 120)

    for _, _, atk_name, _ in ATTACK_SUITE:
        row = [f"{atk_name} Acc"]
        for mode in MODES:
            vals = acc_results[mode][atk_name]
            avg = np.mean(vals) if vals else 0.0
            row.append(f"{avg:.2f}")
        print(f"{row[0]:<20} | {row[1]:<12} | {row[2]:<12} | {row[3]:<12} | {row[4]:<12}")

    print("-" * 120)
    
    f_row = ["FID (vs Real)"]
    l_row = ["LPIPS (vs Cover)"]
    b_row = ["BRISQUE"]
    
    for mode in MODES:
        f_row.append(f"{fid_scores[mode]:.4f}")
        vals = qual_results[mode]
        l_avg = np.mean([v[0] for v in vals]) if vals else 0.0
        b_avg = np.mean([v[1] for v in vals]) if vals else 0.0
        l_row.append(f"{l_avg:.4f}")
        b_row.append(f"{b_avg:.2f}")
    
    print(f"{f_row[0]:<20} | {f_row[1]:<12} | {f_row[2]:<12} | {f_row[3]:<12} | {f_row[4]:<12}")
    print(f"{l_row[0]:<20} | {l_row[1]:<12} | {l_row[2]:<12} | {l_row[3]:<12} | {l_row[4]:<12}")
    print(f"{b_row[0]:<20} | {b_row[1]:<12} | {b_row[2]:<12} | {b_row[3]:<12} | {b_row[4]:<12}")
    print("="*120)

if __name__ == "__main__":
    main()