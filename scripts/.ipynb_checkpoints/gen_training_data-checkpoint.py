# 檔案位置: scripts/gen_training_data.py
import os
import sys
import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from torch import autocast
from tqdm import tqdm

# 路徑設定
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR)) # 加入專案根目錄

# 導入您的模組
try:
    from ldm.util import instantiate_from_config
    from ldm.models.diffusion.dpm_solver import DPMSolverSampler
    import mapping_module 
except ImportError as e:
    print(f"❌ 導入失敗: {e}")
    print("請確保您在 mas_GRDH/scripts 目錄下，且上一層目錄包含 ldm 和 mapping_module")
    sys.exit(1)

# === 設定 ===
# 請根據您的環境修改以下路徑
CKPT_PATH = "/home/vcpuser/netdrive/Workspace/st/mas_GRDH/weights/v1-5-pruned.ckpt"
CONFIG_PATH = os.path.join(os.path.dirname(CURRENT_DIR), "configs/stable-diffusion/ldm.yaml")
OUTPUT_ROOT = os.path.join(os.path.dirname(CURRENT_DIR), "training_data")

# 生成數量 (每類各 500 張，總共 1000 張)
NUM_SAMPLES = 500 
PROMPTS = ["A photo of a landscape", "A cute cat", "A futuristic city", "Delicious food", "Abstract art"]

def load_model(config_path, ckpt_path, device):
    config = OmegaConf.load(config_path)
    # 【修正】加入 weights_only=False 以解決 PyTorch 2.6+ 的反序列化錯誤
    pl_sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[System] 使用設備: {device}")
    
    if not os.path.exists(CKPT_PATH):
        print(f"❌ 找不到權重檔: {CKPT_PATH}")
        return

    model = load_model(CONFIG_PATH, CKPT_PATH, device)
    sampler = DPMSolverSampler(model)
    mapper = mapping_module.ours_mapping(bits=1) # 假設 bit_num=1

    # 建立資料夾
    os.makedirs(os.path.join(OUTPUT_ROOT, "cover"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_ROOT, "stego"), exist_ok=True)

    print(f"🚀 開始生成訓練數據 (目標: {NUM_SAMPLES} 對)...")

    for i in tqdm(range(NUM_SAMPLES)):
        prompt = np.random.choice(PROMPTS)
        seed = np.random.randint(0, 1000000)
        
        # 1. 準備條件
        c = model.get_learned_conditioning([prompt])
        uc = model.get_learned_conditioning([""])
        shape = (4, 512 // 8, 512 // 8)
        
        # === 生成 Cover (隨機噪聲) ===
        # 完全隨機的噪聲，不含秘密
        np.random.seed(seed)
        noise_cover = torch.randn(1, *shape).to(device)
        
        with torch.no_grad(), autocast("cuda"):
            z_0_cover, _ = sampler.sample(
                steps=20, conditioning=c, batch_size=1, shape=shape,
                unconditional_guidance_scale=5.0, unconditional_conditioning=uc,
                x_T=noise_cover # Cover 使用純隨機噪聲
            )
            x_cover = model.decode_first_stage(z_0_cover)
            x_cover = torch.clamp((x_cover + 1.0) / 2.0, min=0.0, max=1.0)
        
        # 保存 Cover
        img_cover = Image.fromarray((x_cover[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
        img_cover.save(os.path.join(OUTPUT_ROOT, "cover", f"{i:05d}.png"))

        # === 生成 Stego (隱寫噪聲) ===
        # 使用 Mapping Module 生成帶秘密的噪聲
        seed_key = np.random.randint(0, 999999)
        # 這裡秘密訊息長度需對應 latent shape，根據您的 mapping_module 實作調整
        # 假設 latent shape 是 (1, 4, 64, 64)
        secret_msg = np.random.randint(0, 2, (1, 4, 64, 64)) 
        
        # 調用您的映射函數生成 Latent
        z_stego_np = mapper.encode_secret(
            secret_message=secret_msg, 
            ori_sample=np.random.randn(1, 4, 64, 64), # 這裡也給一個隨機底
            seed_kernel=seed_key, 
            seed_shuffle=seed_key+123
        )
        noise_stego = torch.from_numpy(z_stego_np).float().to(device)
        
        with torch.no_grad(), autocast("cuda"):
            z_0_stego, _ = sampler.sample(
                steps=20, conditioning=c, batch_size=1, shape=shape,
                unconditional_guidance_scale=5.0, unconditional_conditioning=uc,
                x_T=noise_stego # Stego 使用映射後的噪聲
            )
            x_stego = model.decode_first_stage(z_0_stego)
            x_stego = torch.clamp((x_stego + 1.0) / 2.0, min=0.0, max=1.0)

        # 保存 Stego
        img_stego = Image.fromarray((x_stego[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
        img_stego.save(os.path.join(OUTPUT_ROOT, "stego", f"{i:05d}.png"))

    print("✅ 數據生成完成！")

if __name__ == "__main__":
    main()