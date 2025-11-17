import sys
import os
import argparse
import torch
import numpy as np
from omegaconf import OmegaConf
from torch import autocast
from contextlib import nullcontext
from PIL import Image
from reedsolo import RSCodec 

# === 路徑設定 ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PARENT_DIR)

try:
    from ldm.util import instantiate_from_config
    from ldm.models.diffusion.dpm_solver import DPMSolverSampler
    import mapping_module
    # 移除 hamming_encode_stream 的導入
except ImportError as e:
    print(f"❌ [Alice] 導入失敗: {e}")
    sys.exit(1)

def load_model_from_config(config, ckpt, device):
    print(f"[Alice] 正在載入模型: {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser(description="Alice: 雙模態隱寫系統 - 圖像生成端 (Repetition(3) + RS(255, 119))")
    
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--secret_key", type=int, required=True)
    parser.add_argument("--outpath", type=str, default="outputs/alice_output.png")
    
    parser.add_argument("--ckpt", type=str, default="/home/vcpuser/netdrive/Workspace/stt/mas_GRDH/weights/v1-5-pruned.ckpt")
    parser.add_argument("--config", type=str, default="configs/stable-diffusion/ldm.yaml")
    parser.add_argument("--dpm_steps", type=int, default=50)
    parser.add_argument("--dpm_order", type=int, default=2)
    parser.add_argument("--scale", type=float, default=5.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--bit_num", type=int, default=1)
    opt = parser.parse_args()

    device = torch.device(opt.device)
    if not os.path.exists(opt.config):
        print(f"❌ [Alice] 找不到設定檔: {opt.config}")
        sys.exit(1)

    config = OmegaConf.load(opt.config)
    model = load_model_from_config(config, opt.ckpt, device)
    sampler = DPMSolverSampler(model)

    # === 1. 準備金鑰 ===
    seed_kernel = opt.secret_key
    seed_shuffle = (opt.secret_key + 9527) % (2**32)
    print(f"[Alice] 使用金鑰: {opt.secret_key} (Kernel: {seed_kernel}, Shuffle: {seed_shuffle})")
    
    mapper = mapping_module.ours_mapping(bits=opt.bit_num)
    latent_shape = (1, 4, 64, 64) 

    # === 1.5 【Hybrid ECC 編碼】 RS(255, 119) + Repetition(3) ===
    print("[Alice] 正在執行 Hybrid ECC 編碼 (RS(255, 119) -> Repetition(3))...")
    
    # --- 外層碼：RS(255, 119) ---
    N_ECC_SYMBOLS = 136 # (t=68 符號)
    N_DATA_BYTES_PER_BLOCK = 119
    
    # 【關鍵修正】: 容量必須小於 16384 / 3
    NUM_BLOCKS = 2 
    
    PAYLOAD_SIZE_BYTES = NUM_BLOCKS * N_DATA_BYTES_PER_BLOCK # 2 * 119 = 238 字節
    
    rsc = RSCodec(N_ECC_SYMBOLS)
    rng = np.random.RandomState(opt.secret_key)
    original_secret_bytes = rng.bytes(PAYLOAD_SIZE_BYTES)
    
    # [Step 1: Outer Code (RS Encode)]
    encoded_bytes_list = []
    for i in range(NUM_BLOCKS):
        chunk = original_secret_bytes[i*N_DATA_BYTES_PER_BLOCK : (i+1)*N_DATA_BYTES_PER_BLOCK]
        encoded_chunk = rsc.encode(chunk)
        encoded_bytes_list.append(encoded_chunk)
    encoded_bytes = b"".join(encoded_bytes_list) # 2 * 255 = 510 字節
    rs_coded_bits = np.unpackbits(np.frombuffer(encoded_bytes, dtype=np.uint8)) # 4080 比特

    # [Step 2: Inner Code (Repetition Encode)]
    REPETITION_FACTOR = 3
    hybrid_coded_bits = np.repeat(rs_coded_bits, REPETITION_FACTOR) # 4080 * 3 = 12240 比特
    
    # [Step 3: 映射準備]
    ENCODED_SIZE_BITS = len(hybrid_coded_bits) # 12240
    latent_capacity = np.prod(latent_shape) * opt.bit_num # 16384
    
    print(f"[Alice] 原始數據: {PAYLOAD_SIZE_BYTES} bytes") # 238 bytes
    print(f"[Alice] RS 編碼後: {len(rs_coded_bits)} bits")
    print(f"[Alice] Repetition(3) 編碼後: {ENCODED_SIZE_BITS} bits")
    print(f"[Alice] 潛在空間容量: {latent_capacity} bits")

    if ENCODED_SIZE_BITS > latent_capacity:
        print(f"❌ [Alice] 錯誤：容量超限！")
        sys.exit(1) 
        
    secret_msg_payload = np.zeros(latent_shape, dtype=np.uint8).flatten()
    secret_msg_payload[:ENCODED_SIZE_BITS] = hybrid_coded_bits
    
    # 將剩餘空間填滿隨機比特（這有助於隱蔽性，避免大量 0）
    rng_pad = np.random.RandomState(seed=seed_kernel+1) # 使用不同種子
    random_padding = rng_pad.randint(0, 2**opt.bit_num, latent_capacity - ENCODED_SIZE_BITS)
    secret_msg_payload[ENCODED_SIZE_BITS:] = random_padding
    
    secret_msg = secret_msg_payload.reshape(latent_shape).astype(np.int8)

    # === 2. 加密：將 (Hybrid ECC) 訊息映射為潛在噪聲 z_T ===
    print("[Alice] 正在加密圖像訊息 (ours_mapping)...")
    z_T_np = mapper.encode_secret(
        secret_message=secret_msg, 
        seed_kernel=seed_kernel, 
        seed_shuffle=seed_shuffle
    )
    z_T = torch.from_numpy(z_T_np.astype(np.float32)).to(device)

    # === 3. 生成：使用 DPM-Solver 進行去噪 (內容不變) ===
    # ... (sampler.sample 內容不變) ...
    print(f"[Alice] 開始生成圖像... Prompt 開頭: \"{opt.prompt[:50]}...\"")
    c = model.get_learned_conditioning([opt.prompt])
    uc = model.get_learned_conditioning([""]) if opt.scale != 1.0 else None
    
    with torch.no_grad(), autocast("cuda"):
        z_0, _ = sampler.sample(
            steps=opt.dpm_steps,
            conditioning=c,
            batch_size=1,
            shape=(4, 64, 64),
            unconditional_guidance_scale=opt.scale,
            unconditional_conditioning=uc,
            eta=0.0,
            order=opt.dpm_order,
            x_T=z_T,
            DPMencode=False,
            DPMdecode=True
        )
        x_samples = model.decode_first_stage(z_0)
        
    # === 4. 後處理與儲存 ===
    # ... (儲存邏輯不變) ...
    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
    x_samples = x_samples.cpu().permute(0, 2, 3, 1).numpy()
    image = Image.fromarray((x_samples[0] * 255).astype(np.uint8))
    
    os.makedirs(os.path.dirname(opt.outpath), exist_ok=True)
    image.save(opt.outpath)
    print(f"✅ [Alice] 隱寫圖像已成功儲存至: {opt.outpath}")
    
    # 移除 pad_len.npy，因為我們不再使用 Hamming
    pad_len_path = opt.outpath + ".pad_len.npy"
    if os.path.exists(pad_len_path):
        os.remove(pad_len_path)
        
    np.save(opt.outpath + ".original_secret.npy", np.frombuffer(original_secret_bytes, dtype=np.uint8))

if __name__ == "__main__":
    main()