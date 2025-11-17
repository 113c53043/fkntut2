import os
import sys
import torch
import subprocess
import time
import numpy as np
import re
import shutil 
from collections import defaultdict # 【新增】導入 defaultdict

# === 1. 路徑設定 (模仿 alice_gen.py / bob_extract.py) ===
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR) # 獲取上一層目錄 (即 MAS_GRDH_PATH)

# 【關鍵】將上一層目錄加入 sys.path
sys.path.append(PARENT_DIR) 

# === 2. 導入模組 ===
try:
    from robust_eval import identity, storage, resize, jpeg, mblur, gblur, awgn
    from text_stego_module.stego import TextStegoSystem
    from utils import load_512
    print("✅ [System] 成功導入所有必要的模組 (robust_eval, text_stego_module, utils)")
except ImportError as e:
    print(f"❌ [System] 導入模組失敗: {e}")
    print("請確保: ")
    print(f"  1. robust_eval_main.py, robust_eval.py, utils.py 都在 'scripts' 文件夾中。")
    print(f"  2. 'text_stego_module' 文件夾在 'scripts' 的上一層目錄中。")
    sys.exit(1)

# === 3. 路徑設定 (使用 PARENT_DIR 作為根目錄) ===
MAS_GRDH_PATH = PARENT_DIR 

# 【路徑修正】請確保這些路徑對您當前的環境是正確的
CKPT_PATH = "/home/vcpuser/netdrive/Workspace/st/mas_GRDH/weights/v1-5-pruned.ckpt"
GPT2_PATH = "/nfs/Workspace/st/mas_GRDH/gpt2"
CONFIG_PATH = os.path.join(MAS_GRDH_PATH, "configs/stable-diffusion/ldm.yaml")

# 【修改】: 這是 txt2img.py 需要的 prompt 列表
PROMPT_FILE_LIST = os.path.join(MAS_GRDH_PATH, "text_prompt_dataset", "test_dataset.txt")
TXT2IMG_SCRIPT = os.path.join(MAS_GRDH_PATH, "scripts", "txt2img.py")

ALICE_SCRIPT = os.path.join(MAS_GRDH_PATH, "scripts", "alice_gen.py")
BOB_SCRIPT = os.path.join(MAS_GRDH_PATH, "scripts", "bob_extract.py")
OUTPUT_DIR = os.path.join(MAS_GRDH_PATH, "outputs", "robust_test_results")

# === 4. 定義魯棒性測試套件 ===
# (您可以隨意擴展這裡的因子)
ATTACK_SUITE = [
    (identity, [None], "1_Identity_Control", ".png"),
    (storage, [None], "2_Storage_Save_Load", ".png"),
    (jpeg, [95, 80, 60, 40], "3_JPEG_Compression", ".jpg"),
    (resize, [0.9, 0.7, 0.5], "4_Resize", ".png"),
    (mblur, [3, 5], "5_Median_Blur", ".png"),         
    (gblur, [3, 5], "6_Gaussian_Blur", ".png"),       
    (awgn, [0.01, 0.05, 0.10], "7_Gaussian_Noise", ".png"), 
]

# === 5. 輔助函數 (run_alice_once, run_bob_once, run_txt2img_test) ===
# (這三個函數與上一版相同，不變)

def run_alice_once(text_sys, prompt, session_key, clean_stego_path):
    # (修改：接收 clean_stego_path 作為參數)
    print(f"\n--- [Alice Base Run] Key: {session_key} Prompt: '{prompt[:50]}...' ---")
    
    try:
        stego_prompt_text, _ = text_sys.alice_encode(prompt, session_key)
    except Exception as e:
        print(f"❌ [Alice] 文本編碼失敗: {e}")
        return None, None
    cmd_alice = [
        sys.executable, ALICE_SCRIPT,
        "--prompt", stego_prompt_text,
        "--secret_key", str(session_key),
        "--outpath", clean_stego_path,
        "--ckpt", CKPT_PATH,
        "--config", CONFIG_PATH,
        "--dpm_steps", "50"
    ]
    try:
        subprocess.run(cmd_alice, check=True, cwd=MAS_GRDH_PATH, capture_output=True, text=True, timeout=300)
    except subprocess.CalledProcessError as e:
        print(f"❌ Alice 圖像生成失敗:\n{e.stderr}")
        return None, None
    except subprocess.TimeoutExpired:
        print("❌ Alice 圖像生成超時。")
        return None, None
    print(f"✅ [Alice] 乾淨的隱寫圖像已生成: {clean_stego_path}")
    return clean_stego_path, stego_prompt_text

def run_bob_once(img_path, stego_prompt_text, session_key):
    """
    在指定的圖像上執行一次 Bob 提取流程。
    返回一個包含準確率的字符串 (例如 "100.00%" 或 "80.50%")
    """
    cmd_bob = [
        sys.executable, BOB_SCRIPT,
        "--img_path", img_path,
        "--prompt", stego_prompt_text,
        "--secret_key", str(session_key),
        "--ckpt", CKPT_PATH,
        "--config", CONFIG_PATH,
        "--dpm_steps", "50"
    ]
    try:
        result_bob = subprocess.run(cmd_bob, check=True, cwd=MAS_GRDH_PATH, capture_output=True, text=True, timeout=300)
        
        match = re.search(r"📊 Payload Byte Accuracy: (\d+\.\d+)%", result_bob.stdout)
        if match:
            return f"{match.group(1)}%" # 返回 "XX.XX%"
        if "N/A (No Ground Truth)" in result_bob.stdout:
            print("⚠️ [Bob] 找不到 .npy 驗證文件 (這不應該發生)")
            return "N/A (No .npy)"
        print("[Bob STDOUT DUMP]:\n" + result_bob.stdout[-500:]) 
        return "0.0% (Parse Fail)"
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Bob 圖像提取失敗:\n{e.stderr[-1000:]}")
        match = re.search(r"📊 Payload Byte Accuracy: (\d+\.\d+)%", e.stdout)
        if match:
            return f"{match.group(1)}% (Exec Fail)"
        return "0.0% (Exec Fail)"
    except subprocess.TimeoutExpired:
        print("❌ Bob 圖像提取超時。")
        return "0.0% (Timeout)"

def run_txt2img_test(attack_name_str, factor, single_prompt_file_path):
    """
    執行 txt2img.py (純圖像隱寫) 測試並返回準確率字符串
    """
    attack_map = {
        "1_Identity_Control": "identity",
        "2_Storage_Save_Load": "storage",
        "3_JPEG_Compression": "jpeg",
        "4_Resize": "resize",
        "5_Median_Blur": "mblur",
        "6_Gaussian_Blur": "gblur",
        "7_Gaussian_Noise": "awgn"
    }
    if attack_name_str not in attack_map:
        return "N/A (Attack N/A)"
    
    attack_arg = attack_map[attack_name_str]
    factor_arg = str(factor) if factor is not None else "0.0" 
    
    cmd_txt2img = [
        sys.executable, TXT2IMG_SCRIPT,
        "--ckpt", CKPT_PATH,
        "--config", CONFIG_PATH, 
        "--dpm_steps", "20", 
        "--dpm_order", "2",
        "--scale", "5.0",
        "--test_prompts", single_prompt_file_path, 
        "--attack_layer", attack_arg,
        "--attack_factor", factor_arg,
        "--mapping_func", "ours_mapping",
        "--seed", "42",
    ]
    
    try:
        # cwd 必須是 CURRENT_DIR (即 scripts/)
        result_txt2img = subprocess.run(cmd_txt2img, check=True, cwd=CURRENT_DIR, 
                                        capture_output=True, text=True, timeout=600) 
        
        output = result_txt2img.stdout
        
        match = re.search(r"average accuracy: (\d+\.\d+)", output) 
        if match:
            try:
                acc_ratio = float(match.group(1)) # 獲取 "0.9903..."
                accuracy = f"{acc_ratio * 100:.2f}%" # 轉換為 "99.04%"
                return accuracy
            except Exception as e:
                print(f"❌ txt2img 解析浮點數失敗: {e}")
                return "0.0% (Float Parse Fail)"
        else:
            print(f"[txt2img DUMP]: {output[-500:]}")
            return "0.0% (Regex Parse Fail)"
            
    except subprocess.CalledProcessError as e:
        print(f"❌ txt2img 執行失敗:\n{e.stderr[-1000:]}")
        return "0.0% (Exec Fail)"
    except subprocess.TimeoutExpired:
        print("❌ txt2img 執行超時。")
        return "0.0% (Timeout)"
    except Exception as e:
        print(f"❌ txt2img 發生未知錯誤: {e}")
        return "0.0% (Unknown Fail)"

# === 6. 【重構】的主測試循環 ===

def main():
    print("🚀 魯棒性 (Robustness) 系統性測試腳本啟動 🚀")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- 系統檢查 ---
    if not os.path.exists(GPT2_PATH) or not os.path.exists(CKPT_PATH):
        print(f"❌ [System] 找不到 GPT-2 ({GPT2_PATH}) 或 CKPT ({CKPT_PATH})")
        sys.exit(1)
    if not os.path.exists(TXT2IMG_SCRIPT):
        print(f"❌ [System] 找不到 txt2img.py 腳本: {TXT2IMG_SCRIPT}")
        sys.exit(1)
    if not os.path.exists(PROMPT_FILE_LIST):
        print(f"❌ [System] 找不到 prompt 測試文件: {PROMPT_FILE_LIST}")
        prompts_to_test = ["A beautiful landscape painting"]
    else:
        with open(PROMPT_FILE_LIST, 'r', encoding='utf-8') as f:
            prompts_to_test = [line.strip() for line in f if line.strip()]
        print(f"✅ [System] 成功加載 {len(prompts_to_test)} 個 prompts 進行測試。")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[System] 使用設備: {device}")
    
    text_sys = TextStegoSystem(model_name=GPT2_PATH)

    results_summary = defaultdict(lambda: ([], []))
    
    # --- 循環測試所有 Prompts ---
    for i, base_prompt in enumerate(prompts_to_test):
        print("\n" + "="*60)
        print(f"🔬 正在執行 Prompt #{i+1}/{len(prompts_to_test)}: '{base_prompt[:60]}...'")
        print("="*60)
        
        base_session_key = int(np.random.randint(10000000, 99999999))
        
        # --- [Step A] 執行 Alice 一次 ---
        clean_stego_path = os.path.join(OUTPUT_DIR, f"prompt_{i:03d}_clean_stego.png")
        
        clean_stego_path_result, stego_prompt_text = run_alice_once(text_sys, base_prompt, base_session_key, clean_stego_path)
        
        if not clean_stego_path_result:
            print("❌ [Fatal] 產生乾淨的隱寫圖像失敗，跳過此 prompt。")
            continue

        ground_truth_npy_path = clean_stego_path + ".original_secret.npy"
        if not os.path.exists(ground_truth_npy_path):
             print(f"❌ [Fatal] Alice 未產生驗證文件: {ground_truth_npy_path}")
             continue
        
        # --- [Step B] 為 txt2img.py 創建單一 prompt 臨時文件 ---
        single_prompt_file_path = os.path.join(OUTPUT_DIR, f"prompt_{i:03d}_single_prompt.txt")
        try:
            with open(single_prompt_file_path, 'w', encoding='utf-8') as f:
                f.write(base_prompt + "\n")
        except Exception as e:
            print(f"❌ [Fatal] 創建單一 prompt 測試文件失敗: {e}")
            continue

        # --- [Step C] 加載乾淨的圖像以準備攻擊 ---
        try:
            clean_samples_tensor = load_512(clean_stego_path).to(device)
        except Exception as e:
            print(f"❌ [Fatal] 使用 load_512 加載乾淨圖像失敗: {e}")
            continue

        # --- [Step D] 循環執行所有攻擊 ---
        for attack_func, factors, attack_name, file_ext in ATTACK_SUITE:
            for factor in factors:
                
                # --- 【關鍵修正】: 將 'N/A' 改為 'NA' ---
                factor_str = str(factor) if factor is not None else 'NA' 
                attack_key = f"{attack_name} (Factor: {factor_str})"
                
                print(f"\n--- [TEST] 攻擊: {attack_key} ---")
                
                # 使用修復後的 factor_str
                attacked_img_base_path = os.path.join(OUTPUT_DIR, f"prompt_{i:03d}_attacked_{attack_name}_{factor_str}")
                # --- 【修正結束】 ---
                
                # [Attack]
                try:
                    attack_func(
                        clean_samples_tensor.clone(), 
                        factor, 
                        tmp_image_name=attacked_img_base_path
                    )
                except Exception as e:
                    print(f"❌ [Attack] 應用攻擊 {attack_name} 失敗: {e}")
                    results_summary[attack_key][0].append(0.0) # 記錄 0%
                    results_summary[attack_key][1].append(0.0) # 記錄 0%
                    continue

                bob_target_path = f"{attacked_img_base_path}{file_ext}"
                if not os.path.exists(bob_target_path):
                    print(f"❌ [Attack] 攻擊函數未按預期創建文件: {bob_target_path}")
                    continue

                bob_expected_npy_path = bob_target_path + ".original_secret.npy"
                try:
                    shutil.copyfile(ground_truth_npy_path, bob_expected_npy_path)
                except Exception as e:
                    print(f"❌ [System] 複製驗證文件失敗: {e}")
                    continue

                # --- 測試 1: 雙模態 (Dual-Modal) 系統 (Alice/Bob) ---
                dual_modal_acc_str = run_bob_once(bob_target_path, stego_prompt_text, base_session_key)
                print(f"  [RESULT 1/2] 雙模態 (Ours): {dual_modal_acc_str}")
                
                # --- 測試 2: 純圖像 (Image-Only) 系統 (txt2img.py) ---
                txt2img_acc_str = run_txt2img_test(attack_name, factor, single_prompt_file_path)
                print(f"  [RESULT 2/2] 純圖像 (txt2img.py): {txt2img_acc_str}")

                # --- 儲存結果 (浮點數) ---
                try:
                    results_summary[attack_key][0].append(float(dual_modal_acc_str.replace('%', '')))
                    results_summary[attack_key][1].append(float(txt2img_acc_str.replace('%', '')))
                except (ValueError, TypeError):
                    results_summary[attack_key][0].append(0.0) # 處理 "N/A" 或 "Parse Fail"
                    results_summary[attack_key][1].append(0.0)
                
                time.sleep(1) 

    # --- [Step E] 打印最終的平均報告 ---
    print("\n" + "="*85)
    print(f"📊 魯棒性測試最終報告 (在 {len(prompts_to_test)} 個 Prompts 上的平均結果)")
    print("="*85)
    
    print(f"{'Attack Type & Factor'.ljust(35)} | {'Dual-Modal (Avg. Payload Acc.)'.ljust(30)} | {'Image-Only (Avg. Raw Bit Acc.)'.ljust(25)}")
    print("-" * 90)
    
    # 為了排序，重新遍歷 ATTACK_SUITE
    for _, factors, attack_name, _ in ATTACK_SUITE:
        for factor in factors:
            # --- 【關鍵修正】: 將 'N/A' 改為 'NA' ---
            factor_str = str(factor) if factor is not None else 'NA'
            attack_key = f"{attack_name} (Factor: {factor_str})"
            
            dual_modal_results, txt2img_results = results_summary[attack_key]
            
            if not dual_modal_results: # 如果一次都沒跑成功
                avg_dual_modal = "N/A"
                avg_txt2img = "N/A"
            else:
                avg_dual_modal = f"{np.mean(dual_modal_results):.2f}%"
                avg_txt2img = f"{np.mean(txt2img_results):.2f}%"

            print(f"{attack_key.ljust(35)} | {avg_dual_modal.ljust(30)} | {avg_txt2img.ljust(25)}")
    
    print("="*85)
    print("✅ 系統性測試完成。")

if __name__ == "__main__":
    main()