import os
import sys
import subprocess
import re
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from collections import defaultdict

# === 1. 核心配置 ===
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
MAS_GRDH_PATH = CURRENT_DIR
sys.path.append(MAS_GRDH_PATH)

# 模型與權重
CKPT_PATH = "weights/v1-5-pruned.ckpt"
CONFIG_PATH = os.path.join(MAS_GRDH_PATH, "configs/stable-diffusion/ldm.yaml")

# 腳本路徑 (確保指向您修正後的 Fixed 版本)
ALICE_SCRIPT = os.path.join(MAS_GRDH_PATH, "pure_alice_uncertainty_fixed.py")
BOB_SCRIPT = os.path.join(MAS_GRDH_PATH, "pure_bob.py")

# 輸出路徑
OUTPUT_DIR = os.path.join(MAS_GRDH_PATH, "outputs", "tuning_comprehensive")
PAYLOAD_FILE = os.path.join(OUTPUT_DIR, "payload.dat")
PROMPT_FILE = os.path.join(MAS_GRDH_PATH, "text_prompt_dataset", "test_dataset.txt")

# === 2. 參數實驗組 (您可以自由調整這裡) ===
EXPERIMENTS = [
    {"lr": "0.25", "reg": "0.05", "name": "1. High Robust (Baseline)"},
    {"lr": "0.20", "reg": "0.2",  "name": "2. Mid-High"},
    {"lr": "0.15", "reg": "0.4",  "name": "3. Balanced"},
    {"lr": "0.10", "reg": "0.8",  "name": "4. Quality Focused"},
    {"lr": "0.05", "reg": "1.5",  "name": "5. Ultra Quality"}
]

# 設定為您想要的新總數
NUM_SAMPLES = 50 

# === 3. 輔助函數 ===

def load_prompts(limit=10):
    default_prompt = "A futuristic city skyline, cinematic lighting, 8k"
    if os.path.exists(PROMPT_FILE):
        with open(PROMPT_FILE, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
        while len(prompts) < limit:
            prompts.extend(prompts)
        return prompts[:limit]
    else:
        return [default_prompt] * limit

def ensure_paths():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(PAYLOAD_FILE):
        with open(PAYLOAD_FILE, "wb") as f:
            f.write(os.urandom(2048))

def get_image_diff(path_a, path_b):
    try:
        img_a = np.array(Image.open(path_a).convert("RGB")).astype(np.float32)
        img_b = np.array(Image.open(path_b).convert("RGB")).astype(np.float32)
        return np.mean(np.abs(img_a - img_b))
    except:
        return 999.0

def run_subprocess(cmd):
    try:
        result = subprocess.run(
            cmd, check=True, cwd=MAS_GRDH_PATH, 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        return ""

def run_alice_base(prompt, session_key, out_path, latent_path):
    """生成 Baseline (Iter 0)"""
    # 【修正】: 只要圖片存在就跳過，不強制檢查 latent_path (因為傳入的是空字串)
    if os.path.exists(out_path):
        if not latent_path or os.path.exists(latent_path):
            return # Skip generation
    
    cmd = [
        sys.executable, ALICE_SCRIPT,
        "--prompt", prompt, 
        "--secret_key", str(session_key), 
        "--payload_path", PAYLOAD_FILE,
        "--outpath", out_path, 
        "--ckpt", CKPT_PATH, "--config", CONFIG_PATH,
        "--opt_iters", "0", 
        "--init_latent", "" 
    ]
    run_subprocess(cmd)

def run_alice_exp(prompt, session_key, out_path, lr, reg):
    # 只要圖片存在就跳過 (這會讓前 44 張秒過)
    if os.path.exists(out_path): return
    
    cmd = [
        sys.executable, ALICE_SCRIPT,
        "--prompt", prompt, 
        "--secret_key", str(session_key), 
        "--payload_path", PAYLOAD_FILE,
        "--outpath", out_path, 
        "--ckpt", CKPT_PATH, "--config", CONFIG_PATH,
        "--opt_iters", "10", 
        "--lr", str(lr), 
        "--lambda_reg", str(reg),
        "--use_uncertainty"
    ]
    run_subprocess(cmd)

def run_bob(img_path, prompt, session_key):
    # Bob 必須跑，因為我們需要重新收集準確率數據來畫表
    # 但 Bob 跑得比 Alice 快很多
    cmd = [
        sys.executable, BOB_SCRIPT,
        "--img_path", img_path, 
        "--prompt", prompt, 
        "--secret_key", str(session_key), 
        "--gt_path", PAYLOAD_FILE,
        "--ckpt", CKPT_PATH, "--config", CONFIG_PATH
    ]
    output = run_subprocess(cmd)
    match = re.search(r"Bit Accuracy.*?: (\d+\.\d+)%", output)
    if match: return float(match.group(1))
    return 0.0

# === 4. 主程式 ===

def main():
    ensure_paths()
    print(f"🚀 Comprehensive Parameter Tuning (Target: {NUM_SAMPLES}) 🚀")
    print(f"   (Will skip generation for existing {NUM_SAMPLES} images)")
    
    prompts = load_prompts(NUM_SAMPLES)
    stats = defaultdict(lambda: {'acc': [], 'diff': [], 'config': {}})
    vis_paths = {} 

    for i, prompt in tqdm(enumerate(prompts), total=NUM_SAMPLES, desc="Processing Images"):
        session_key = 100000 + i
        
        # 1. Baseline
        base_name = f"p{i}_base.png"
        base_path = os.path.join(OUTPUT_DIR, base_name)
        run_alice_base(prompt, session_key, base_path, "")
        
        if i == 0: vis_paths['base'] = base_path

        # 2. Experiments
        for exp_idx, exp in enumerate(EXPERIMENTS):
            exp_name = f"p{i}_exp{exp_idx}.png"
            exp_path = os.path.join(OUTPUT_DIR, exp_name)
            
            # 若檔案已存在，這裡會瞬間跳過
            run_alice_exp(prompt, session_key, exp_path, exp['lr'], exp['reg'])
            
            # Bob 還是要跑一下來讀取準確率
            acc = run_bob(exp_path, prompt, session_key)
            diff = get_image_diff(exp_path, base_path)
            
            stats[exp_idx]['acc'].append(acc)
            stats[exp_idx]['diff'].append(diff)
            stats[exp_idx]['config'] = exp
            
            if i == 0: vis_paths[exp_idx] = exp_path

    # === 5. 生成報告 ===
    print("\n" + "="*80)
    print(f"{'Config Name'.ljust(25)} | {'LR'.ljust(6)} | {'Reg'.ljust(6)} | {'Avg Acc'.ljust(10)} | {'Avg Diff'.ljust(10)} | {'Score'}")
    print("-" * 80)
    
    best_score = -999
    best_idx = -1
    
    results = []

    for idx, data in stats.items():
        avg_acc = np.mean(data['acc'])
        avg_diff = np.mean(data['diff'])
        cfg = data['config']
        
        if avg_acc < 99.0:
            score = -9999 + avg_acc
            mark = "❌ Unstable"
        else:
            score = -avg_diff 
            mark = "✅ Pass"
            
        if score > best_score:
            best_score = score
            best_idx = idx
            
        print(f"{cfg['name'].ljust(25)} | {cfg['lr'].ljust(6)} | {cfg['reg'].ljust(6)} | {f'{avg_acc:.2f}%'.ljust(10)} | {f'{avg_diff:.2f}'.ljust(10)} | {mark}")
        
        results.append({
            "idx": idx, "name": cfg['name'], "lr": cfg['lr'], "reg": cfg['reg'],
            "avg_acc": avg_acc, "avg_diff": avg_diff, "path": vis_paths[idx]
        })

    print("="*80)
    print(f"🏆 Best Configuration: {EXPERIMENTS[best_idx]['name']}")
    print(f"   (LR={EXPERIMENTS[best_idx]['lr']}, Reg={EXPERIMENTS[best_idx]['reg']})")

    # === 6. 生成視覺化大圖 ===
    if 'base' in vis_paths and os.path.exists(vis_paths['base']):
        img_base = Image.open(vis_paths['base']).convert("RGB")
        w, h = img_base.size
        
        grid_w = w * len(results)
        grid_h = h * 2 + 120
        canvas = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
        except:
            font = None

        canvas.paste(img_base, (0, 60))
        draw.text((10, 10), "0. Baseline (Target)\n(No Optimization)", fill="black", font=font)

        for i, res in enumerate(results):
            if not os.path.exists(res['path']): continue
            img = Image.open(res['path']).convert("RGB")
            x_offset = i * w
            
            canvas.paste(img, (x_offset, 60))
            
            is_best = (res['idx'] == best_idx)
            color = "green" if res['avg_acc'] > 99.0 else "red"
            prefix = "🏆 " if is_best else ""
            
            info = f"{prefix}{res['name']}\nLR:{res['lr']} Reg:{res['reg']}\nAvg Acc: {res['avg_acc']:.2f}%\nAvg Diff: {res['avg_diff']:.2f}"
            draw.text((x_offset + 10, 10), info, fill=color, font=font)

        final_path = os.path.join(OUTPUT_DIR, "Final_Comprehensive_Report.png")
        canvas.save(final_path)
        print(f"\n🖼️  Visual report saved to: {final_path}")

if __name__ == "__main__":
    main()