import os
import sys
import subprocess
import numpy as np
import re
import shutil 
import torch
from collections import defaultdict
import json
import time
from tqdm import tqdm
import cv2
import lpips
import matplotlib.pyplot as plt
import random

# 嘗試引用 BRISQUE
try:
    from piq import brisque
    BRISQUE_AVAILABLE = True
except ImportError:
    BRISQUE_AVAILABLE = False
    print("⚠️ PIQ not found. BRISQUE will be skipped.")

# === 1. 路徑設定 ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR) 
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

try:
    from robust_eval import awgn, jpeg, identity
    from utils import load_512
except ImportError:
    sys.path.append(os.path.join(PARENT_DIR, "scripts"))
    from robust_eval import awgn, jpeg, identity
    from utils import load_512

MAS_GRDH_PATH = PARENT_DIR 
CKPT_PATH = "/home/vcpuser/netdrive/Workspace/stt/mas_GRDH/weights/v1-5-pruned.ckpt"
if not os.path.exists(CKPT_PATH):
    CKPT_PATH = os.path.join(MAS_GRDH_PATH, "weights/v1-5-pruned.ckpt")
CONFIG_PATH = os.path.join(MAS_GRDH_PATH, "configs/stable-diffusion/ldm.yaml")
PROMPT_FILE_LIST = os.path.join(MAS_GRDH_PATH, "text_prompt_dataset", "coco_dataset.txt")
ALICE_SCRIPT = os.path.join(MAS_GRDH_PATH, "pure_alice_final.py")
BOB_SCRIPT = os.path.join(MAS_GRDH_PATH, "pure_bob.py")

OUTPUT_DIR = os.path.join(MAS_GRDH_PATH, "outputs", "boundary_search") # 獨立輸出目錄
DIR_COVER = os.path.join(OUTPUT_DIR, "cover")

# === 2. 實驗配置 (邊界突破) ===
# 保持 N=50 以確保統計穩定性
TOTAL_SAMPLES = 50 

# 參數網格：專注於高強度區間
# 包含原本的邊界 0.12，並大幅向上探索
LR_LIST = [0.12, 0.14, 0.16, 0.18, 0.20] 

# 鎖定在表現最好的 Reg 區間 (縮小範圍以節省時間)
REG_LIST = [1.0, 1.25]

# 代表性攻擊
PROXY_ATTACKS = [
    (awgn, [0.05], "Noise(0.05)", ".png"),
    (jpeg, [50], "JPEG(50)", ".jpg")
]

# === 3. 輔助函式 ===
def get_subprocess_env():
    env = os.environ.copy()
    python_path = env.get("PYTHONPATH", "")
    new_paths = [PARENT_DIR, os.path.join(PARENT_DIR, "scripts")]
    for p in new_paths:
        if p not in python_path:
            python_path = f"{p}{os.pathsep}{python_path}"
    env["PYTHONPATH"] = python_path
    return env

def create_gt_bits_file(payload_path, out_gt_path):
    CAPACITY_BYTES = 16384 // 8
    with open(payload_path, "rb") as f: raw_data = f.read()
    payload_data = raw_data
    if len(payload_data) > CAPACITY_BYTES - 2: payload_data = payload_data[:CAPACITY_BYTES-2]
    length_header = len(payload_data).to_bytes(2, 'big')
    final_payload = length_header + payload_data
    if len(final_payload) < CAPACITY_BYTES: final_payload += b'\x00' * (CAPACITY_BYTES - len(final_payload))
    np.save(out_gt_path, np.frombuffer(final_payload, dtype=np.uint8))

def run_alice_generation(prompt, session_key, out_path, payload_path, lr, reg):
    cmd = [
        sys.executable, ALICE_SCRIPT,
        "--prompt", prompt, "--secret_key", str(session_key),
        "--payload_path", payload_path, "--outpath", out_path,
        "--ckpt", CKPT_PATH, "--config", CONFIG_PATH,
        "--opt_iters", "10", "--dpm_steps", "20",
        "--mode", "adaptive", 
        "--lr", str(lr), "--lambda_reg", str(reg)
    ]
    try:
        subprocess.run(cmd, check=True, cwd=MAS_GRDH_PATH, capture_output=True, text=True, timeout=300, env=get_subprocess_env())
        return True
    except:
        return False

def run_bob_decode(img_path, prompt, session_key):
    cmd = [
        sys.executable, BOB_SCRIPT, "--img_path", img_path,
        "--prompt", prompt, "--secret_key", str(session_key),
        "--ckpt", CKPT_PATH, "--config", CONFIG_PATH, "--dpm_steps", "20"
    ]
    try:
        res = subprocess.run(cmd, check=True, cwd=MAS_GRDH_PATH, capture_output=True, text=True, timeout=120, env=get_subprocess_env())
        match = re.search(r"Bit Accuracy.*?: (\d+\.\d+)%", res.stdout)
        return float(match.group(1)) if match else 0.0
    except: return 0.0

class QualityEvaluator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lpips_fn = lpips.LPIPS(net='alex').to(self.device)
        
    def calc_lpips(self, p1, p2):
        try:
            t1 = self._load(p1); t2 = self._load(p2)
            with torch.no_grad(): return self.lpips_fn(t1, t2).item()
        except: return 0.0
        
    def calc_brisque(self, p1):
        if not BRISQUE_AVAILABLE: return 0.0
        try:
            t1 = self._load(p1, norm=False) 
            with torch.no_grad(): 
                return brisque(t1, data_range=1.0).item()
        except: return 0.0

    def _load(self, p, norm=True):
        img = cv2.imread(p); img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (512,512)) / 255.0
        if norm: img = img * 2 - 1
        return torch.tensor(img.transpose(2,0,1)).float().to(self.device).unsqueeze(0)

def calculate_pareto_frontier(results):
    sorted_results = sorted(results, key=lambda x: x['brisque'])
    frontier = []
    current_max_acc = -1.0
    for r in sorted_results:
        if r['acc'] > current_max_acc:
            frontier.append(r)
            current_max_acc = r['acc']
    return frontier

def find_optimal_knee_point(results):
    frontier = calculate_pareto_frontier(results)
    if not frontier: return None
    accs = np.array([r['acc'] for r in frontier])
    brisques = np.array([r['brisque'] for r in frontier])
    if len(accs) > 1:
        norm_acc = (accs - accs.min()) / (accs.max() - accs.min() + 1e-8)
        norm_bri = (brisques - brisques.min()) / (brisques.max() - brisques.min() + 1e-8)
    else:
        norm_acc, norm_bri = accs, brisques
    # Distance to (Acc=1, Brisque=0)
    distances = np.sqrt((1.0 - norm_acc)**2 + (0.0 - norm_bri)**2)
    best_idx = np.argmin(distances)
    return frontier[best_idx]

def plot_pareto_analysis(results, output_path="boundary_pareto.png"):
    accs = [r['acc'] for r in results]
    brisques = [r['brisque'] for r in results]
    lrs = [r['lr'] for r in results]
    regs = [r['reg'] for r in results]
    
    frontier = calculate_pareto_frontier(results)
    frontier_x = [r['brisque'] for r in frontier]
    frontier_y = [r['acc'] for r in frontier]
    best_knee = find_optimal_knee_point(results)

    plt.figure(figsize=(12, 8))
    # 畫出所有點，顏色代表 LR 強度
    scatter = plt.scatter(brisques, accs, c=lrs, cmap='plasma', s=100, alpha=0.8, edgecolors='k')
    cbar = plt.colorbar(scatter)
    cbar.set_label('Learning Rate (LR)')

    plt.plot(frontier_x, frontier_y, 'r--', linewidth=2, alpha=0.7, label='Pareto Frontier')
    
    if best_knee:
        plt.scatter([best_knee['brisque']], [best_knee['acc']], 
                    c='gold', s=250, marker='*', edgecolors='k', zorder=10, 
                    label=f"Sweet Spot\nLR={best_knee['lr']}, Reg={best_knee['reg']}")

    for i, (acc, bri) in enumerate(zip(accs, brisques)):
        # 只標示 Reg，讓圖面乾淨點
        plt.annotate(f"R{regs[i]}", (bri, acc), fontsize=8, xytext=(5, 5), textcoords='offset points')

    plt.title(f'Boundary Search: Pushing the Limits (N={TOTAL_SAMPLES})')
    plt.xlabel('BRISQUE Score (Lower is Better)')
    plt.ylabel('Average Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    return best_knee

# === 4. 主流程 ===
def main():
    print(f"🚀 BOUNDARY SEARCH (N={TOTAL_SAMPLES}) 🚀")
    print(f"Goal: Testing LR > 0.12 to find the collapse point.")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DIR_COVER, exist_ok=True)
    
    payload_path = os.path.join(OUTPUT_DIR, "payload.dat")
    with open(payload_path, "wb") as f: f.write(os.urandom(2048))
    
    prompts = []
    if os.path.exists(PROMPT_FILE_LIST):
        with open(PROMPT_FILE_LIST) as f: lines = [l.strip() for l in f if l.strip()]
        while len(prompts) < TOTAL_SAMPLES: prompts.extend(lines)
        prompts = prompts[:TOTAL_SAMPLES]
    else:
        prompts = ["A futuristic city"] * TOTAL_SAMPLES

    evaluator = QualityEvaluator()
    results = [] 

    # Phase 0: 預先生成 Cover
    print("\n[Phase 0] Generating Covers...")
    for i in tqdm(range(TOTAL_SAMPLES), desc="Covers"):
        cover_p = os.path.join(DIR_COVER, f"{i:05d}.png")
        if not os.path.exists(cover_p):
            run_alice_generation(prompts[i], 10000+i, cover_p, payload_path, lr=0.05, reg=1.5)

    total_combos = len(LR_LIST) * len(REG_LIST)
    curr = 0

    # Phase 1: 網格搜索
    for lr in LR_LIST:
        for reg in REG_LIST:
            curr += 1
            print(f"\n[{curr}/{total_combos}] Testing LR={lr}, Reg={reg} ...")
            
            combo_dir = os.path.join(OUTPUT_DIR, f"lr{lr}_reg{reg}")
            os.makedirs(combo_dir, exist_ok=True)
            
            acc_scores = []
            brisque_scores = []
            
            for i in tqdm(range(TOTAL_SAMPLES), desc="Sampling", leave=False):
                session_key = 10000 + i
                prompt = prompts[i]
                cover_p = os.path.join(DIR_COVER, f"{i:05d}.png")
                stego_p = os.path.join(combo_dir, f"{i:05d}.png")
                gt_p = stego_p + ".gt_bits.npy"
                
                # 生成
                if not os.path.exists(stego_p):
                    success = run_alice_generation(prompt, session_key, stego_p, payload_path, lr, reg)
                    if not success: continue
                    create_gt_bits_file(payload_path, gt_p)
                
                # 評估
                if os.path.exists(stego_p):
                    brisque_val = evaluator.calc_brisque(stego_p)
                    brisque_scores.append(brisque_val)
                
                # 攻擊
                try:
                    img_tensor = load_512(stego_p).cuda()
                    sample_accs = []
                    for atk_fn, args, atk_name, ext in PROXY_ATTACKS:
                        att_p = stego_p.replace(".png", f"_{atk_name}.png").replace(".png", ext)
                        atk_fn(img_tensor.clone(), args[0], tmp_image_name=att_p.replace(ext, ""))
                        shutil.copyfile(gt_p, att_p + ".gt_bits.npy")
                        acc = run_bob_decode(att_p, prompt, session_key)
                        sample_accs.append(acc)
                        if os.path.exists(att_p): os.remove(att_p)
                        if os.path.exists(att_p + ".gt_bits.npy"): os.remove(att_p + ".gt_bits.npy")
                    
                    if sample_accs:
                        acc_scores.append(np.mean(sample_accs)) 
                    del img_tensor
                except: pass
            
            if acc_scores:
                avg_acc = np.mean(acc_scores)
                avg_brisque = np.mean(brisque_scores)
                print(f"   -> Acc: {avg_acc:.2f}%, BRISQUE: {avg_brisque:.2f}")
                results.append({"lr": lr, "reg": reg, "acc": avg_acc, "brisque": avg_brisque})
            
            shutil.rmtree(combo_dir)

    # Phase 2: 報告
    print("\n" + "="*80)
    print("BOUNDARY SEARCH RESULTS")
    print(f"{'LR':<8} | {'Reg':<8} | {'Acc':<10} | {'BRISQUE':<10}")
    print("-" * 80)
    
    results.sort(key=lambda x: x['acc'], reverse=True)
    for r in results:
        print(f"{r['lr']:<8} | {r['reg']:<8} | {r['acc']:<10.2f} | {r['brisque']:<10.2f}")
    
    print("="*80)
    
    # 全域最佳解更新
    best_knee = find_optimal_knee_point(results)
    if best_knee:
        print(f"\n🏆 New Global Optimal (Knee Point):")
        print(f"   LR={best_knee['lr']}, Reg={best_knee['reg']}")
        print(f"   Acc={best_knee['acc']:.2f}%, BRISQUE={best_knee['brisque']:.2f}")

    plot_pareto_analysis(results, os.path.join(OUTPUT_DIR, "boundary_pareto.png"))

if __name__ == "__main__":
    main()