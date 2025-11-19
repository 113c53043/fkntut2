import sys
import os
import argparse
import torch
import numpy as np
from omegaconf import OmegaConf
from torch import autocast
from PIL import Image
from reedsolo import RSCodec, ReedSolomonError

# === 路徑設定 ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR)

try:
    from ldm.util import instantiate_from_config
    from ldm.models.diffusion.dpm_solver import DPMSolverSampler
    import mapping_module
except ImportError as e:
    print(f"❌ [Bob] 導入失敗: {e}")
    sys.exit(1)

def load_model_from_config(config, ckpt, device):
    pl_sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    return model

def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    if w != 512 or h != 512: 
        image = image.resize((512, 512), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    return torch.from_numpy(2.*image - 1.)

def repetition_decode_soft(soft_bits, rep_factor=3):
    """
    軟判決重複碼解碼器
    """
    # 確保長度是 rep_factor 的倍數
    if len(soft_bits) % rep_factor != 0:
        pad_len = rep_factor - (len(soft_bits) % rep_factor)
        soft_bits = np.pad(soft_bits, (0, pad_len), 'constant', constant_values=0.5)
        
    # 將 [0.1, 0.8, 0.3, 0.9, 0.8, 0.7] 重塑為 [[0.1, 0.8, 0.3], [0.9, 0.8, 0.7]]
    grouped_bits = soft_bits.reshape(-1, rep_factor)
    
    # 對每一組取平均值
    averaged_bits = np.mean(grouped_bits, axis=1)
    
    # 對平均值進行最終的硬判決
    hard_bits = np.round(averaged_bits).astype(np.uint8)
    
    return hard_bits

def calc_bit_accuracy(bytes_a, bytes_b):
    """
    計算兩個 bytes 對象之間的位元準確率 (Bit Accuracy)
    """
    # 轉為整數列表以便操作
    arr_a = np.frombuffer(bytes_a, dtype=np.uint8)
    arr_b = np.frombuffer(bytes_b, dtype=np.uint8)
    
    # 處理長度不一致 (取交集部分，多出的算錯)
    min_len = min(len(arr_a), len(arr_b))
    max_len = max(len(arr_a), len(arr_b))
    
    # 比較交集部分
    # XOR 運算：相同為0，不同為1
    xor_diff = arr_a[:min_len] ^ arr_b[:min_len]
    
    # 計算 XOR 結果中 '1' 的個數 (即錯誤的 bit 數)
    # np.unpackbits 將 uint8 展開為 8 個 bit (0/1)
    bit_diffs = np.unpackbits(xor_diff).sum()
    
    # 加上長度差異部分的 bit (全部視為錯誤)
    len_diff_bytes = max_len - min_len
    total_bit_errors = bit_diffs + (len_diff_bytes * 8)
    
    total_bits = max_len * 8
    if total_bits == 0: return 100.0
    
    accuracy = (1 - (total_bit_errors / total_bits)) * 100.0
    return accuracy

def main():
    parser = argparse.ArgumentParser(description="Bob: 雙模態隱寫系統 - 圖像提取端 (Repetition(3) + RS(255, 119))")
    
    parser.add_argument("--img_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--secret_key", type=int, required=True)
    
    parser.add_argument("--ckpt", type=str, default="/home/vcpuser/netdrive/Workspace/GRDH_ORG/weights/sd-v1-5.ckpt")
    parser.add_argument("--config", type=str, default="configs/stable-diffusion/ldm.yaml")
    parser.add_argument("--dpm_steps", type=int, default=50)
    parser.add_argument("--dpm_order", type=int, default=2)
    parser.add_argument("--scale", type=float, default=5.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--bit_num", type=int, default=1)
    opt = parser.parse_args()

    device = torch.device(opt.device)
    config = OmegaConf.load(opt.config)
    model = load_model_from_config(config, opt.ckpt, device)
    sampler = DPMSolverSampler(model)

    # === 1. 準備解密金鑰 ===
    seed_kernel = opt.secret_key
    seed_shuffle = (opt.secret_key + 9527) % (2**32) 
    print(f"[Bob] 使用金鑰解密: {opt.secret_key}")

    # === 2. 圖像反演 (DPM Inversion) ===
    init_image = load_img(opt.img_path).to(device)
    c = model.get_learned_conditioning([opt.prompt])
    uc = model.get_learned_conditioning([""]) if opt.scale != 1.0 else None
    with torch.no_grad(), autocast("cuda"):
        init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))
        z_T_hat, _ = sampler.sample(
            steps=opt.dpm_steps,
            conditioning=c,
            batch_size=1,
            shape=init_latent.shape[1:],
            unconditional_guidance_scale=opt.scale,
            unconditional_conditioning=uc,
            eta=0.0,
            order=opt.dpm_order,
            x_T=init_latent,
            DPMencode=True,
            DPMdecode=False
        )

    # === 3. 解密訊息 (Soft Decoding) ===
    mapper = mapping_module.ours_mapping(bits=opt.bit_num)
    
    recovered_soft_array = mapper.decode_secret_soft(
        pred_noise=z_T_hat.cpu().numpy(),
        seed_kernel=seed_kernel,
        seed_shuffle=seed_shuffle
    )

    # === 3.5 【Hybrid ECC 解碼】 Repetition(3) -> RS(255, 119) ===
    print("[Bob] 正在執行 Hybrid ECC 解碼 (Repetition(3) -> RS(255, 119))...")
    
    # 恢復 RS 碼的參數
    N_ECC_SYMBOLS = 136 
    N_DATA_BYTES_PER_BLOCK = 119
    NUM_BLOCKS = 2 
    BLOCK_SIZE = 255
    PAYLOAD_SIZE_BYTES = NUM_BLOCKS * N_DATA_BYTES_PER_BLOCK # 238 字節
    
    # --- [Step 1: Inner Code (Repetition Decode)] ---
    rs_bits_len = NUM_BLOCKS * BLOCK_SIZE * 8 # 4080
    REPETITION_FACTOR = 3
    hybrid_bits_len = rs_bits_len * REPETITION_FACTOR # 12240
    
    recovered_soft_bits = recovered_soft_array.flatten()[:hybrid_bits_len]
    
    # **實際調用 Repetition 軟判決解碼**
    rs_coded_bits_fixed = repetition_decode_soft(recovered_soft_bits, rep_factor=REPETITION_FACTOR)
    print(f"[Bob] 內層碼 (Repetition(3)) 已完成軟判決。")

    # --- [Step 2: Outer Code (RS Decode)] ---
    rsc = RSCodec(N_ECC_SYMBOLS)
    bit_padding = (8 - (len(rs_coded_bits_fixed) % 8)) % 8
    if bit_padding == 8: bit_padding = 0
    rs_coded_bits_padded = np.pad(rs_coded_bits_fixed, (0, bit_padding), 'constant', constant_values=0)
    
    try:
        recovered_bytes_with_ecc = np.packbits(rs_coded_bits_padded).tobytes()
    except Exception as e:
        print(f"❌ [Bob] packbits 失敗: {e}.")
        sys.exit(1)
    
    repaired_bytes_list = []
    total_rs_fixed = 0
    ecc_fail_count = 0
    
    for i in range(NUM_BLOCKS):
        chunk = recovered_bytes_with_ecc[i*BLOCK_SIZE : (i+1)*BLOCK_SIZE]
        if len(chunk) < BLOCK_SIZE:
             chunk += b'\x00' * (BLOCK_SIZE - len(chunk))

        try:
            repaired_chunk, errors_fixed, errata_pos = rsc.decode(chunk)
            if len(errata_pos) > 0: 
                total_rs_fixed += len(errata_pos)
            repaired_bytes_list.append(repaired_chunk)
        except ReedSolomonError: 
            ecc_fail_count += 1
            # 如果解碼失敗，保留原始（經過Repetition修正但RS失敗的）數據，以便計算 Bit Accuracy
            repaired_bytes_list.append(chunk[:N_DATA_BYTES_PER_BLOCK])
    
    print(f"[Bob] 外層碼 (RS(255, 119)) 修正了 {total_rs_fixed} 個字節錯誤。")
    if ecc_fail_count > 0:
        print(f"⚠️ [Bob] 警告： {ecc_fail_count} 個 RS 區塊解碼失敗！(錯誤超過 t=68)")
        
    final_repaired_bytes = b"".join(repaired_bytes_list)
    final_repaired_bytes = final_repaired_bytes[:PAYLOAD_SIZE_BYTES] 

    # === 4. 驗證與輸出 ===
    secret_path = opt.img_path + ".original_secret.npy"
    if os.path.exists(secret_path):
        gt_bytes = np.load(secret_path).tobytes()
        
        # 計算 Bit Accuracy
        bit_acc = calc_bit_accuracy(final_repaired_bytes, gt_bytes)
        
        if final_repaired_bytes == gt_bytes:
            print("="*30)
            print("🎉 驗證成功！資訊完整還原！")
            # 輸出 Bit Accuracy 供 robust_eval_main.py 抓取
            print(f"📊 Payload Bit Accuracy: 100.00%")
            print("="*30)
        else:
            total_bytes = len(gt_bytes)
            mismatched_bytes = sum(1 for a, b in zip(final_repaired_bytes, gt_bytes) if a != b)
            if len(final_repaired_bytes) != len(gt_bytes):
                mismatched_bytes += abs(len(final_repaired_bytes) - len(gt_bytes))
            
            # Byte Accuracy (僅供參考)
            byte_accuracy = ((total_bytes - mismatched_bytes) / total_bytes) * 100.0 if total_bytes > 0 else 0.0

            print("="*30)
            print(f"❌ 驗證失敗：最終數據不匹配！")
            print(f"  - 總字節: {total_bytes}")
            print(f"  - 錯誤字節: {mismatched_bytes} (Byte Acc: {byte_accuracy:.2f}%)")
            # 輸出 Bit Accuracy 供 robust_eval_main.py 抓取
            print(f"📊 Payload Bit Accuracy: {bit_acc:.2f}%")
            print("="*30)
    else:
        print("✅ [Bob] 解密完成 (無原始檔可供比對)")
        print(f"⚠️ [Bob] 找不到驗證文件: {secret_path}")
        print(f"📊 Payload Bit Accuracy: N/A (No Ground Truth)")

if __name__ == "__main__":
    main()