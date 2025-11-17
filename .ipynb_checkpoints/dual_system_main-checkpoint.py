import os
import sys
import torch
import subprocess
import time
import numpy as np
import re

# === 全域路徑設定 ===
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__)) 
MAS_GRDH_PATH = CURRENT_DIR
TEXT_MODULE_PATH = os.path.join(CURRENT_DIR, 'text_stego_module')

# 【路徑修正】使用您 GitHub 上的路徑
CKPT_PATH = "/home/vcpuser/netdrive/Workspace/stt/mas_GRDH/weights/v1-5-pruned.ckpt"
# 【路徑修正】假設您的 gpt2 位於項目根目錄下的 'gpt2'
GPT2_PATH = os.path.join(MAS_GRDH_PATH, "gpt2") 
CONFIG_PATH = os.path.join(MAS_GRDH_PATH, "configs/stable-diffusion/ldm.yaml")
PROMPT_FILE_LIST = os.path.join(MAS_GRDH_PATH, "text_prompt_dataset", "test_dataset.txt")

ALICE_SCRIPT = os.path.join(MAS_GRDH_PATH, "scripts", "alice_gen.py")
BOB_SCRIPT = os.path.join(MAS_GRDH_PATH, "scripts", "bob_extract.py")
OUTPUT_DIR = os.path.join(MAS_GRDH_PATH, "outputs", "batch_test")

# 加入模組路徑
sys.path.append(MAS_GRDH_PATH)
try:
    from text_stego_module.stego import TextStegoSystem
    print("✅ [System] 文本模組載入成功")
except ImportError:
    print(f"❌ [System] 找不到文本模組 (text_stego_module)，請確認目錄結構。路徑: {TEXT_MODULE_PATH}")
    sys.exit(1)

def ensure_paths():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(PROMPT_FILE_LIST):
        print(f"⚠️ 警告：找不到測試 Prompt 文件: {PROMPT_FILE_LIST}")
        print("將使用預設 prompts...")
        return ["A futuristic city with flying cars", "A cute cat sitting on a bench"]
    with open(PROMPT_FILE_LIST, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines() if line.strip()]

def run_single_experiment(text_sys, prompt, session_key, idx):
    print(f"\n--- [Experiment #{idx:03d}] Key: {session_key} ---")
    
    stego_img_path = os.path.join(OUTPUT_DIR, f"exp_{idx:03d}.png")
    
    # [Step 1] Alice: 文本隱寫
    try:
        stego_prompt_text, generated_ids = text_sys.alice_encode(prompt, session_key)
    except Exception as e:
        print(f"❌ [Alice] 文本編碼失敗: {e}")
        return False, 0.0

    # [Step 2] Alice: 圖像隱寫
    cmd_alice = [
        sys.executable, ALICE_SCRIPT,
        "--prompt", stego_prompt_text,
        "--secret_key", str(session_key),
        "--outpath", stego_img_path,
        "--ckpt", CKPT_PATH,
        "--config", CONFIG_PATH,
        "--dpm_steps", "50"
    ]
    try:
        result_alice = subprocess.run(cmd_alice, check=True, cwd=MAS_GRDH_PATH, capture_output=True, text=True, timeout=300)
    except subprocess.CalledProcessError as e:
        # 【關鍵修正】: 同時打印 stdout 和 stderr
        print(f"❌ Alice 圖像生成失敗:")
        print("--- ALICE STDOUT ---")
        print(e.stdout)
        print("--- ALICE STDERR ---")
        print(e.stderr)
        print("--------------------")
        return False, 0.0
    except subprocess.TimeoutExpired:
        print("❌ Alice 圖像生成超時。")
        return False, 0.0

    # [Step 3] Bob: 文本提取
    try:
        extracted_key = text_sys.bob_decode(generated_ids)
    except Exception as e:
        print(f"❌ [Bob] 文本解碼失敗: {e}")
        return False, 0.0
        
    if extracted_key != session_key:
        print(f"❌ 文本金鑰提取失敗 (Exp: {session_key}, Got: {extracted_key})")
        return False, 0.0
    print(f"✅ 文本金鑰提取成功: {extracted_key}")

    # [Step 4] Bob: 圖像提取
    cmd_bob = [
        sys.executable, BOB_SCRIPT,
        "--img_path", stego_img_path,
        "--prompt", stego_prompt_text,
        "--secret_key", str(extracted_key),
        "--ckpt", CKPT_PATH,
        "--config", CONFIG_PATH,
        "--dpm_steps", "50"
    ]
    try:
        result_bob = subprocess.run(cmd_bob, check=True, cwd=MAS_GRDH_PATH, capture_output=True, text=True, timeout=300)
    except subprocess.CalledProcessError as e:
        # 【關鍵修正】: 同時打印 stdout 和 stderr
        print(f"❌ Bob 圖像提取失敗:")
        print("--- BOB STDOUT ---")
        print(e.stdout)
        print("--- BOB STDERR ---")
        print(e.stderr)
        print("--------------------")
        return True, 0.0 # 文本成功，圖像失敗
    except subprocess.TimeoutExpired:
        print("❌ Bob 圖像提取超時。")
        return True, 0.0

    # 解析準確率 (基於 bob_extract.py 的標準輸出)
    ecc_success = "🎉 驗證成功" in result_bob.stdout
    
    if ecc_success:
        print(f"✅ 實驗成功！Hybrid ECC 最終還原率: 100%")
    else:
        # 打印 Bob 的日誌以供除錯
        print("⚠️ 實驗成功但 ECC 修復失敗。")
        print("--- Bob STDOUT ---")
        print(result_bob.stdout)
        print("--- Bob STDERR ---")
        print(result_bob.stderr)
        print("--------------------")

    return True, (100.0 if ecc_success else 0.0)

def main():
    num_runs = 1
    if len(sys.argv) > 1:
        try:
            num_runs = int(sys.argv[1])
        except ValueError:
            print("參數錯誤：請輸入一個整數，例如 'python3 dual_system_main.py 50'")
            sys.exit(1)
            
    print(f"\n🚀 雙模態系統 (Hybrid ECC) - 批量測試啟動 (預計執行 {num_runs} 回合) 🚀\n")

    prompts = ensure_paths()
    if not os.path.exists(GPT2_PATH):
        print(f"❌ [System] 找不到 GPT-2 模型路徑: {GPT2_PATH}")
        print("請確保 GPT-2 模型已下載並放置在項目根目錄的 'gpt2' 文件夾中。")
        sys.exit(1)
        
    text_sys = TextStegoSystem(model_name=GPT2_PATH)
    
    total_start = time.time()
    results = []
    
    for i in range(num_runs):
        prompt = prompts[i % len(prompts)]
        session_key = int(np.random.randint(10000000, 99999999))
        
        try:
            text_success, ecc_success_rate = run_single_experiment(text_sys, prompt, session_key, i+1)
            results.append((text_success, ecc_success_rate))
        except Exception as e:
            print(f"❌ [Experiment #{i+1}] 發生嚴重錯誤: {e}")
            results.append((False, 0.0))

    # 統計報告
    total_runs = len(results)
    if total_runs == 0:
        print("沒有執行任何測試。")
        return
        
    text_successful_runs = sum(1 for r in results if r[0])
    ecc_successful_runs = sum(1 for r in results if r[1] == 100.0)
    
    system_text_success_rate = (text_successful_runs / total_runs) * 100
    system_final_success_rate = (ecc_successful_runs / total_runs) * 100

    print("\n" + "="*40)
    print("📊 最終實驗報告 (Final Report)")
    print("="*40)
    print(f"執行總回合數: {total_runs}")
    print(f"文本金鑰成功回合: {text_successful_runs}")
    print(f"圖像ECC修復成功回合: {ecc_successful_runs}")
    print(f"文本金鑰成功率: {system_text_success_rate:.2f}%")
    print(f"系統最終成功率 (End-to-End): {system_final_success_rate:.2f}%")
    print(f"總耗時: {(time.time() - total_start)/60:.2f} 分鐘")
    print("="*40)

if __name__ == "__main__":
    main()