import os
import sys
import torch
import subprocess
import numpy as np
from collections import defaultdict

# === 1. 路徑設定 ===
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR)

# === 2. 導入必要模組 ===
try:
    from text_stego_module.stego import TextStegoSystem
    from scripts.xunet import XuNetEvaluator
    print("✅ [System] 安全性測試模組導入成功")
except ImportError as e:
    print(f"❌ [System] 導入失敗: {e}")
    sys.exit(1)

# === 3. 全域配置 (請確認這些路徑) ===
MAS_GRDH_PATH = PARENT_DIR
CKPT_PATH = "/home/vcpuser/netdrive/Workspace/st/mas_GRDH/weights/v1-5-pruned.ckpt"
GPT2_PATH = "/nfs/Workspace/st/mas_GRDH/gpt2"
CONFIG_PATH = os.path.join(MAS_GRDH_PATH, "configs/stable-diffusion/ldm.yaml")
PROMPT_FILE_LIST = os.path.join(MAS_GRDH_PATH, "text_prompt_dataset", "test_dataset.txt")
ALICE_SCRIPT = os.path.join(MAS_GRDH_PATH, "scripts", "alice_gen.py")

# 【重要】請設定您的 Xu-Net 權重路徑
XUNET_CKPT_PATH = "/nfs/Workspace/stt/mas_GRDH/weights/xunet_best.pth" 

# 輸出目錄 (建議與魯棒性測試分開)
OUTPUT_DIR = os.path.join(MAS_GRDH_PATH, "outputs", "security_test_results")

# === 4. Alice 生成函數 ===
def run_alice_only(text_sys, prompt, session_key, output_path):
    """
    僅執行 Alice 生成隱寫圖像，不進行後續攻擊測試
    """
    try:
        stego_prompt_text, _ = text_sys.alice_encode(prompt, session_key)
    except Exception as e:
        print(f"❌ 文本編碼失敗: {e}")
        return None

    cmd_alice = [
        sys.executable, ALICE_SCRIPT,
        "--prompt", stego_prompt_text,
        "--secret_key", str(session_key),
        "--outpath", output_path,
        "--ckpt", CKPT_PATH,
        "--config", CONFIG_PATH,
        "--dpm_steps", "50"
    ]
    
    try:
        # 執行 Alice 腳本
        subprocess.run(cmd_alice, check=True, cwd=MAS_GRDH_PATH, capture_output=True, text=True, timeout=300)
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"❌ Alice 生成失敗:\n{e.stderr}")
        return None
    except subprocess.TimeoutExpired:
        print("❌ Alice 生成超時")
        return None

# === 5. 主程式 ===
def main():
    print("🛡️ 安全性 (Security Analysis) 獨立測試腳本啟動 🛡️")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 初始化 ---
    if not os.path.exists(PROMPT_FILE_LIST):
        print(f"⚠️ 找不到 Prompt 文件，使用預設測試")
        prompts = ["A fast red car driving on the highway"]
    else:
        with open(PROMPT_FILE_LIST, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
    
    print(f"[System] 加載 {len(prompts)} 個 Prompts 進行測試")
    
    # 初始化 TextStego 和 XuNet
    text_sys = TextStegoSystem(model_name=GPT2_PATH)
    
    if os.path.exists(XUNET_CKPT_PATH):
        security_evaluator = XuNetEvaluator(ckpt_path=XUNET_CKPT_PATH)
    else:
        print(f"⚠️ [Warning] 找不到權重檔 {XUNET_CKPT_PATH}，使用隨機初始化模型進行流程測試。")
        security_evaluator = XuNetEvaluator(ckpt_path=None)

    scores = []
    
    print("\n" + "="*80)
    print(f"{'ID'.ljust(5)} | {'Prompt Preview'.ljust(40)} | {'Xu-Net Score (Prob)'.ljust(20)} | {'Result'}")
    print("-" * 80)

    # --- 測試循環 ---
    for i, prompt in enumerate(prompts):
        prompt_id = f"{i+1:03d}"
        session_key = int(np.random.randint(10000000, 99999999))
        
        # 1. 生成圖片
        stego_img_path = os.path.join(OUTPUT_DIR, f"sec_test_{prompt_id}.png")
        if not run_alice_only(text_sys, prompt, session_key, stego_img_path):
            print(f"{prompt_id}   | 生成失敗".ljust(80))
            continue
            
        # 2. 安全性檢測 (Steganalysis)
        stego_prob = security_evaluator.eval_image(stego_img_path)
        scores.append(stego_prob)
        
        # 判斷結果 (一般而言 0.5 為閾值，越低越好)
        result_str = "✅ Secure" if stego_prob < 0.5 else "⚠️ Detectable"
        prompt_preview = (prompt[:37] + "...") if len(prompt) > 37 else prompt
        
        print(f"{prompt_id}   | {prompt_preview.ljust(40)} | {f'{stego_prob:.4f}'.ljust(20)} | {result_str}")

    # --- 最終報告 ---
    if scores:
        avg_score = sum(scores) / len(scores)
        print("="*80)
        print(f"\n📊 安全性測試總結 (共 {len(scores)} 張圖片):")
        print(f"   平均被偵測機率 (Avg. Stego Probability): {avg_score:.4f}")
        print(f"   (理想目標：接近 0.0 或小於 0.5)")
        
        # 計算 Anti-Steganalysis Accuracy (欺騙率)
        # 即被判定為 Cover (Prob < 0.5) 的比例
        undetected_count = sum(1 for s in scores if s < 0.5)
        detection_accuracy = (1 - (undetected_count / len(scores))) * 100
        print(f"   Xu-Net 偵測成功率: {detection_accuracy:.2f}%")
        print(f"   我方逃逸成功率 (Undetectability): {(undetected_count / len(scores)) * 100:.2f}%")
    else:
        print("❌ 沒有產生有效的測試結果。")

    print("="*80)

if __name__ == "__main__":
    main()