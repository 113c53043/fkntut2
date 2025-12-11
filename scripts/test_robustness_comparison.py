import os
import sys
import subprocess
import numpy as np
import re
import shutil 
import torch
from collections import defaultdict

# === 1. 路徑設定 ===
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR) 
sys.path.append(PARENT_DIR) 

# === 2. 導入攻擊模組 ===
try:
    from robust_eval import identity, storage, resize, jpeg, mblur, gblur, awgn
    from utils import load_512
    print("✅ [System] 成功導入攻擊模組")
except ImportError:
    try:
        from scripts.robust_eval import identity, storage, resize, jpeg, mblur, gblur, awgn
        from scripts.utils import load_512
        print("✅ [System] 成功導入攻擊模組 (Package Import)")
    except ImportError as e:
        print(f"❌ [System] 導入模組失敗: {e}")
        sys.exit(1)

# === 3. 核心配置 ===
MAS_GRDH_PATH = PARENT_DIR 
CKPT_PATH = "/home/vcpuser/netdrive/Workspace/stt/mas_GRDH/weights/v1-5-pruned.ckpt"
if not os.path.exists(CKPT_PATH):
    CKPT_PATH = os.path.join(MAS_GRDH_PATH, "weights/v1-5-pruned.ckpt")

CONFIG_PATH = os.path.join(MAS_GRDH_PATH, "configs/stable-diffusion/ldm.yaml")
PROMPT_FILE_LIST = os.path.join(MAS_GRDH_PATH, "text_prompt_dataset", "test_dataset.txt")

# === 【關鍵修正】指向 _fixed.py 版本，確保邏輯與 FID 測試一致 ===
ALICE_SCRIPT = os.path.join(MAS_GRDH_PATH, "pure_alice_fixed.py")
ALICE_UNC_SCRIPT = os.path.join(MAS_GRDH_PATH, "pure_alice_uncertainty_fixed.py")
BOB_SCRIPT = os.path.join(MAS_GRDH_PATH, "pure_bob.py")
TXT2IMG_SCRIPT = os.path.join(MAS_GRDH_PATH, "scripts", "txt2img.py") 

OUTPUT_DIR = os.path.join(MAS_GRDH_PATH, "outputs", "robust_sota_results")

# === 4. 定義攻擊套件 ===
ATTACK_SUITE = [
    (identity, [None], "1_Identity_Control", ".png"),
    (storage, [None], "2_Storage_Save_Load", ".png"),
    (jpeg, [95, 80, 60, 50], "3_JPEG_Compression", ".jpg"),
    (resize, [0.9, 0.75, 0.5], "4_Resize", ".png"),
    (mblur, [3, 5], "5_Median_Blur", ".png"),            
    (gblur, [3, 5], "6_Gaussian_Blur", ".png"),         
    (awgn, [0.01, 0.05], "7_Gaussian_Noise", ".png"), 
]

# === 5. 輔助函數 ===

def run_alice_generic(script_path, prompt, session_key, out_path, payload_path, extra_args=[]):
    """
    通用 Alice 執行函數
    """
    cmd_alice = [
        sys.executable, script_path,
        "--prompt", prompt, 
        "--secret_key", str(session_key),
        "--payload_path", payload_path,
        "--outpath", out_path,
        "--ckpt", CKPT_PATH,
        "--config", CONFIG_PATH,
        "--opt_iters", "10",
        "--dpm_steps", "20"
        # 注意：LR 和 Reg 現在透過 extra_args 傳入，不再寫死
    ] + extra_args
    
    try:
        # 隱藏大量輸出，只顯示錯誤
        result = subprocess.run(cmd_alice, check=True, cwd=MAS_GRDH_PATH, capture_output=True, text=True, timeout=600)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Alice Crash ({os.path.basename(script_path)}):\n{e.stderr}")
        return False
    
def run_bob_once(img_path, prompt, session_key, gt_path):
    cmd_bob = [
        sys.executable, BOB_SCRIPT,
        "--img_path", img_path,
        "--prompt", prompt,
        "--secret_key", str(session_key),
        "--gt_path", gt_path,
        "--ckpt", CKPT_PATH,
        "--config", CONFIG_PATH,
        "--dpm_steps", "20"
    ]
    
    try:
        result_bob = subprocess.run(cmd_bob, check=True, cwd=MAS_GRDH_PATH, capture_output=True, text=True, timeout=300)
        match = re.search(r"Bit Accuracy.*?: (\d+\.\d+)%", result_bob.stdout)
        if match: return f"{match.group(1)}%"
        return "0.00%"
    except: return "0.00%"

def run_txt2img_test(attack_name_str, factor, single_prompt_file_path):
    attack_map = {
        "1_Identity_Control": "identity", "2_Storage_Save_Load": "storage",
        "3_JPEG_Compression": "jpeg", "4_Resize": "resize",
        "5_Median_Blur": "mblur", "6_Gaussian_Blur": "gblur",
        "7_Gaussian_Noise": "awgn"
    }
    if attack_name_str not in attack_map: return "N/A"
    
    cmd_txt2img = [
        sys.executable, TXT2IMG_SCRIPT,
        "--ckpt", CKPT_PATH, "--config", CONFIG_PATH, 
        "--dpm_steps", "20", "--scale", "5.0",
        "--test_prompts", single_prompt_file_path, 
        "--attack_layer", attack_map[attack_name_str], 
        "--attack_factor", str(factor) if factor is not None else "0.0",
        "--seed", "42", "--quiet"
    ]
    try:
        result = subprocess.run(cmd_txt2img, check=True, cwd=CURRENT_DIR, capture_output=True, text=True, timeout=600)
        match = re.search(r"average accuracy: (\d+\.\d+)", result.stdout)
        if match: return f"{float(match.group(1)) * 100:.2f}%"
    except: pass
    return "0.00%"

def parse_percentage(val_str):
    try: return float(val_str.replace('%', '').split(' ')[0])
    except: return None

# === 6. 主程式 ===

def main():
    print("🚀 Robustness Comparison (SOTA Params): Pure vs. Uncertainty vs. Baseline 🚀")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(PROMPT_FILE_LIST):
        prompts_to_test = ["A futuristic city skyline, cinematic lighting, 8k"]
    else:
        with open(PROMPT_FILE_LIST, 'r', encoding='utf-8') as f:
            prompts_to_test = [line.strip() for line in f if line.strip()]

    results_summary = defaultdict(lambda: ([], [], []))
    
    # 測試 20 張即可看出趨勢
    prompts_to_test = prompts_to_test[:20] 
    
    for i, base_prompt in enumerate(prompts_to_test):
        print(f"\n🔬 Prompt #{i+1}: '{base_prompt[:40]}...'")
        session_key = int(np.random.randint(10000000, 99999999))
        payload_path = os.path.join(OUTPUT_DIR, f"p{i}_payload.dat")
        
        pure_stego_path = os.path.join(OUTPUT_DIR, f"p{i}_pure_stego.png")
        pure_gt_bits = pure_stego_path + ".gt_bits.npy"
        
        unc_stego_path = os.path.join(OUTPUT_DIR, f"p{i}_unc_stego.png")
        unc_gt_bits = unc_stego_path + ".gt_bits.npy"
        
        with open(payload_path, "wb") as f: f.write(os.urandom(2048))
            
        # 2. 執行 Pure Alice (原始參數)
        # LR=0.25 (高強度，對照組)
        print("  ⚡ Running Pure Alice (LR=0.25)...")
        success_pure = run_alice_generic(ALICE_SCRIPT, base_prompt, session_key, pure_stego_path, payload_path, 
                                         extra_args=["--lr", "0.25"])
        
        # 3. 執行 Uncertainty Alice (SOTA 參數)
        # LR=0.05, Reg=1.5 (我們調優出的最佳畫質參數)
        print("  ⚡ Running Ours Unc (LR=0.05, Reg=1.5)...")
        success_unc = run_alice_generic(
            ALICE_UNC_SCRIPT, base_prompt, session_key, unc_stego_path, payload_path, 
            extra_args=["--use_uncertainty", "--lr", "0.05", "--lambda_reg", "1.5"]
        )

        if not success_pure and not success_unc: continue

        # 4. 攻擊測試
        img_tensor_pure = load_512(pure_stego_path).cuda() if success_pure else None
        img_tensor_unc = load_512(unc_stego_path).cuda() if success_unc else None
        
        for attack_func, factors, attack_name, file_ext in ATTACK_SUITE:
            for factor in factors:
                factor_str = str(factor) if factor is not None else 'NA'
                attack_key = f"{attack_name} (Fac: {factor_str})"
                
                # A. Pure
                acc_pure = "N/A"
                if img_tensor_pure is not None:
                    att_path = os.path.join(OUTPUT_DIR, f"p{i}_pure_{attack_name}_{factor_str}")
                    try:
                        attack_func(img_tensor_pure.clone(), factor, tmp_image_name=att_path)
                        final_path = f"{att_path}{file_ext}"
                        shutil.copyfile(pure_gt_bits, final_path + ".gt_bits.npy")
                        acc_pure = run_bob_once(final_path, base_prompt, session_key, payload_path)
                    except: pass

                # B. Ours
                acc_unc = "N/A"
                if img_tensor_unc is not None:
                    att_path = os.path.join(OUTPUT_DIR, f"p{i}_unc_{attack_name}_{factor_str}")
                    try:
                        attack_func(img_tensor_unc.clone(), factor, tmp_image_name=att_path)
                        final_path = f"{att_path}{file_ext}"
                        # 確保 GT bits 存在
                        if os.path.exists(unc_gt_bits):
                            shutil.copyfile(unc_gt_bits, final_path + ".gt_bits.npy")
                        else:
                            shutil.copyfile(pure_gt_bits, final_path + ".gt_bits.npy")
                        acc_unc = run_bob_once(final_path, base_prompt, session_key, payload_path)
                    except: pass

                # C. Baseline
                tmp_prompt_file = os.path.join(OUTPUT_DIR, f"p{i}_tmp.txt")
                if not os.path.exists(tmp_prompt_file):
                    with open(tmp_prompt_file, 'w') as f: f.write(base_prompt)
                acc_base = run_txt2img_test(attack_name, factor, tmp_prompt_file)

                # print(f"  {attack_key}: Pure={acc_pure} | Unc={acc_unc} | Base={acc_base}")

                val_pure = parse_percentage(acc_pure)
                val_unc = parse_percentage(acc_unc)
                val_base = parse_percentage(acc_base)

                if val_pure is not None: results_summary[attack_key][0].append(val_pure)
                if val_unc is not None: results_summary[attack_key][1].append(val_unc)
                if val_base is not None: results_summary[attack_key][2].append(val_base)

    # === 最終統計 ===
    print("\n" + "="*100)
    print(f"{'Attack Condition'.ljust(40)} | {'Pure (Avg)'.ljust(15)} | {'Uncertainty'.ljust(15)} | {'Base (Avg)'.ljust(15)}")
    print("-" * 100)
    
    for _, factors, attack_name, _ in ATTACK_SUITE:
        for factor in factors:
            factor_str = str(factor) if factor is not None else 'NA'
            attack_key = f"{attack_name} (Fac: {factor_str})"
            res = results_summary[attack_key]
            
            avg_pure = f"{np.mean(res[0]):.2f}%" if res[0] else "N/A"
            avg_unc = f"{np.mean(res[1]):.2f}%" if res[1] else "N/A"
            avg_base = f"{np.mean(res[2]):.2f}%" if res[2] else "N/A"
            
            print(f"{attack_key.ljust(40)} | {avg_pure.ljust(15)} | {avg_unc.ljust(15)} | {avg_base}")
            
    print("="*100)

if __name__ == "__main__":
    main()