import os
import sys
import subprocess
import torch
import numpy as np
import re
import cv2
import ssl
import shutil
from collections import defaultdict

# === 0. SSL Context ===
ssl._create_default_https_context = ssl._create_unverified_context

# === 1. 環境設定 ===
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
MAS_GRDH_PATH = PARENT_DIR 
sys.path.append(MAS_GRDH_PATH)

# 檢查 BRISQUE (piq)
try:
    from piq import brisque
    print("✅ [System] piq (BRISQUE) loaded successfully.")
except ImportError:
    print("❌ [System] piq not found! Please run: pip install piq")
    sys.exit(1)

# 路徑設定
ALICE_UNC_SCRIPT = os.path.join(MAS_GRDH_PATH, "pure_alice_uncertainty_fixed.py")
BOB_SCRIPT = os.path.join(MAS_GRDH_PATH, "pure_bob.py")
CKPT_PATH = "weights/v1-5-pruned.ckpt"
CONFIG_PATH = os.path.join(MAS_GRDH_PATH, "configs/stable-diffusion/ldm.yaml")
OUTPUT_DIR = os.path.join(MAS_GRDH_PATH, "outputs", "tuning_balance")
PAYLOAD_PATH = os.path.join(OUTPUT_DIR, "payload.dat")

# === 2. 實驗組設定 ===
CONFIGS = [
    {"name": "Quality (Current)", "lr": "0.05", "reg": "1.5"},
    {"name": "Balanced (Rec)",   "lr": "0.10", "reg": "0.8"},
    {"name": "Robust (High LR)", "lr": "0.15", "reg": "0.4"}
]

PROMPT = "a cute kitten playing soccer, tiny cat kicking a football on a grassy field, dynamic action pose, bright sunlight, soft fur texture"

def ensure_paths():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(PAYLOAD_PATH):
        with open(PAYLOAD_PATH, "wb") as f: f.write(os.urandom(2048))

def run_subprocess(cmd, debug_name=None):
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=MAS_GRDH_PATH)
        return res.stdout, res.stderr
    except Exception as e:
        print(f"Error: {e}")
        return "", str(e)

def calculate_brisque_score(img_path):
    if not os.path.exists(img_path): return 0.0
    try:
        img = cv2.imread(img_path)
        if img is None: return 0.0
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        t_img = torch.tensor(img.transpose(2, 0, 1)).unsqueeze(0).float()
        score = brisque(t_img, data_range=1.0)
        return score.item()
    except Exception as e:
        # print(f"BRISQUE Err: {e}")
        return 0.0

def jpeg_attack(img_path, quality=50):
    if not os.path.exists(img_path): return None
    from PIL import Image
    img = Image.open(img_path)
    out_path = img_path.replace(".png", f"_jpg{quality}.jpg")
    img.save(out_path, "JPEG", quality=quality)
    
    # === 關鍵修正：複製 GT Bits 給 JPEG 檔案 ===
    src_gt = img_path + ".gt_bits.npy"
    dst_gt = out_path + ".gt_bits.npy"
    if os.path.exists(src_gt):
        shutil.copyfile(src_gt, dst_gt)
    else:
        # 如果這裡報錯，代表 Alice 根本沒生成 GT 檔
        print(f"⚠️ Warning: GT bits NOT found at {src_gt}. Alice might have failed.")
        
    return out_path

def parse_acc(stdout, stderr, name):
    # 嘗試抓取準確率
    match = re.search(r"Bit Accuracy.*?: (\d+\.\d+)%", stdout)
    if match:
        return float(match.group(1))
    else:
        # 如果抓不到，代表 Bob 執行失敗或崩潰
        if "traceback" in stderr.lower() or "error" in stderr.lower():
            print(f"\n❌ [Bob Error] on {name}:")
            print(stderr[-500:]) 
        elif stdout:
            # 只有當沒有找到 Acc 且沒有明顯 Error 時才印出輸出
            pass 
        return 0.0

def main():
    ensure_paths()
    print("🚀 Tuning for Balance (Debug Mode) 🚀\n")
    
    results = defaultdict(lambda: {"brisque": [], "acc_clean": [], "acc_jpg50": []})
    
    for i in range(5): 
        session_key = 77777 + i
        print(f"📸 Image {i+1}/5...")
        
        for cfg in CONFIGS:
            out_name = f"img{i}_{cfg['name'].split()[0]}.png"
            out_path = os.path.join(OUTPUT_DIR, out_name)
            gt_path = out_path + ".gt_bits.npy"
            
            # 1. 生成 (Alice)
            # 邏輯修正：只要缺圖或缺GT，就強制刪除重跑，確保資料一致
            if not os.path.exists(out_path) or not os.path.exists(gt_path):
                print(f"   ⚙️ Generating {cfg['name']}...")
                
                # 安全刪除舊檔
                if os.path.exists(out_path): os.remove(out_path)
                if os.path.exists(gt_path): os.remove(gt_path)

                cmd = [
                    sys.executable, ALICE_UNC_SCRIPT,
                    "--prompt", PROMPT, 
                    "--secret_key", str(session_key), 
                    "--payload_path", PAYLOAD_PATH,
                    "--outpath", out_path,
                    "--ckpt", CKPT_PATH, "--config", CONFIG_PATH,
                    "--opt_iters", "10", 
                    "--lr", str(cfg['lr']), 
                    "--lambda_reg", str(cfg['reg']),
                    "--use_uncertainty"
                ]
                out, err = run_subprocess(cmd)
                
                if not os.path.exists(out_path):
                    print(f"   ⚠️ Generation FAILED for {cfg['name']}")
                    # 印出 Alice 的錯誤訊息幫助除錯
                    print(f"   Alice Error Log: {err[-500:]}")
                    continue

            # 2. 畫質評估
            b_score = calculate_brisque_score(out_path)
            results[cfg['name']]["brisque"].append(b_score)
            
            # 3. 魯棒性 (Clean)
            cmd_bob = [sys.executable, BOB_SCRIPT, "--img_path", out_path, "--prompt", PROMPT, 
                       "--secret_key", str(session_key), "--gt_path", PAYLOAD_PATH, "--ckpt", CKPT_PATH, "--config", CONFIG_PATH]
            out, err = run_subprocess(cmd_bob)
            acc_clean = parse_acc(out, err, f"{cfg['name']} (Clean)")
            results[cfg['name']]["acc_clean"].append(acc_clean)
            
            # 4. 魯棒性 (JPEG 50)
            jpg_path = jpeg_attack(out_path, quality=50)
            if jpg_path:
                cmd_bob_jpg = [sys.executable, BOB_SCRIPT, "--img_path", jpg_path, "--prompt", PROMPT, 
                               "--secret_key", str(session_key), "--gt_path", PAYLOAD_PATH, "--ckpt", CKPT_PATH, "--config", CONFIG_PATH]
                out, err = run_subprocess(cmd_bob_jpg)
                acc_jpg = parse_acc(out, err, f"{cfg['name']} (JPEG50)")
                results[cfg['name']]["acc_jpg50"].append(acc_jpg)

    # === 報告 ===
    print("\n" + "="*90)
    print(f"{'Config Name'.ljust(25)} | {'BRISQUE (↓)'.ljust(12)} | {'Acc Clean'.ljust(15)} | {'Acc JPEG 50'.ljust(15)}")
    print("-" * 90)
    
    for cfg in CONFIGS:
        name = cfg['name']
        if not results[name]["brisque"]:
            print(f"{name.ljust(25)} | N/A")
            continue
            
        avg_b = np.mean(results[name]["brisque"])
        avg_clean = np.mean(results[name]["acc_clean"])
        avg_jpg = np.mean(results[name]["acc_jpg50"])
        
        print(f"{name.ljust(25)} | {f'{avg_b:.2f}'.ljust(12)} | {f'{avg_clean:.2f}%'.ljust(15)} | {f'{avg_jpg:.2f}%'.ljust(15)}")
        
    print("="*90)

if __name__ == "__main__":
    main()