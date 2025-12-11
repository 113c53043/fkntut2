import sys
import os
import argparse
import torch
import numpy as np
from omegaconf import OmegaConf
from torch import autocast
from PIL import Image, ImageEnhance

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)
sys.path.append(os.path.join(CURRENT_DIR, "scripts"))

try:
    # 導入新的 Tuned Mapping
    from mapping_module_tuned import ours_mapping_tuned
    from ldm.util import instantiate_from_config
    from ldm.models.diffusion.dpm_solver import DPMSolverSampler
except ImportError:
    pass

def load_model_from_config(config, ckpt, device):
    from ldm.util import instantiate_from_config
    print(f"Loading model from {ckpt}...")
    try:
        pl_sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    except TypeError:
        pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"] if "state_dict" in pl_sd else pl_sd
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    return model

def estimate_uncertainty(model, sampler, z_center, c, uc, scale, device, repeats=4, noise_std=0.05):
    z_recs = []
    fast_steps = 10 
    with torch.no_grad(), autocast("cuda"):
        for i in range(repeats):
            noise = torch.randn_like(z_center) * noise_std
            z_input = z_center + noise
            z_0, _ = sampler.sample(steps=fast_steps, conditioning=c, batch_size=1, shape=(4, 64, 64),
                                    unconditional_guidance_scale=scale, unconditional_conditioning=uc,
                                    x_T=z_input, DPMencode=False, DPMdecode=True, verbose=False)
            z_rec, _ = sampler.sample(steps=fast_steps, conditioning=c, batch_size=1, shape=(4, 64, 64),
                                      unconditional_guidance_scale=scale, unconditional_conditioning=uc,
                                      x_T=z_0, DPMencode=True, DPMdecode=False, verbose=False)
            z_recs.append(z_rec)
    stack = torch.stack(z_recs)
    variance = torch.var(stack, dim=0)
    variance_mean = torch.mean(variance, dim=1, keepdim=True) 
    v_min = variance_mean.min()
    v_max = variance_mean.max()
    norm_var = (variance_mean - v_min) / (v_max - v_min + 1e-8)
    
    # === 軟遮罩 (Soft Mask) ===
    # 避免 Mask 為 0，保證即使在高頻/不確定區域，也有至少 20% 的寫入強度
    mask = 1.0 - norm_var
    mask = torch.pow(mask, 2)
    mask = mask * 0.8 + 0.2 
    return mask.repeat(1, 4, 1, 1)

def apply_refinement(pil_image, enable=False):
    """
    Refinement (Sharpening/Contrast) visually looks good but hurts FID and BRISQUE scores.
    Disabled by default for Paper Reproduction.
    """
    if not enable:
        return pil_image
        
    enhancer = ImageEnhance.Sharpness(pil_image)
    pil_image = enhancer.enhance(1.1) 
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(1.05) 
    return pil_image

def generate_alice_image(model, sampler, prompt, secret_key, payload_data, outpath, init_latent_path=None, 
                         opt_iters=15, use_uncertainty=True, dpm_steps=20, scale=5.0, device="cuda",
                         # 這些參數現在是為了兼容性，內部邏輯會覆蓋它們
                         lr=None, lambda_reg=None): 
    
    # 1. Prepare Latent with Tuned Mapping
    if init_latent_path and os.path.exists(init_latent_path):
        z_target = torch.load(init_latent_path, map_location=device)
    else:
        CAPACITY_BYTES = 16384 // 8 
        bits = np.unpackbits(np.frombuffer(payload_data, dtype=np.uint8))
        if len(bits) < 16384: bits = np.pad(bits, (0, 16384 - len(bits)), 'constant')
        bits = bits[:16384].reshape(1, 4, 64, 64)
        
        # === 使用 Tuned Mapping ===
        # scale=0.9: 降低 10% 噪聲能量，顯著幫助 FID
        # clip_range=3.0: 消除極端值，消除偽影
        mapper = ours_mapping_tuned(bits=1, scale=0.9, clip_range=3.0)
        
        z_target_numpy = mapper.encode_secret(secret_message=bits, seed_kernel=secret_key, seed_shuffle=secret_key + 999)
        z_target = torch.from_numpy(z_target_numpy).float().to(device)

    # 2. Setup Optimization
    negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
    c = model.get_learned_conditioning([prompt])
    uc = model.get_learned_conditioning([negative_prompt])

    if use_uncertainty:
        uncertainty_mask = estimate_uncertainty(model, sampler, z_target, c, uc, scale, device)
    else:
        uncertainty_mask = torch.ones_like(z_target)

    z_opt = z_target.clone()
    z_opt.requires_grad = False 
    z_best = z_target.clone() 
    min_loss = float('inf')

    # === 【關鍵策略】動態調度 (Dynamic Scheduling) ===
    # 這是同時獲得高準確率和低 FID 的核心
    
    split_idx = int(opt_iters * 0.6) # 前 60% 為注入期

    for i in range(opt_iters + 1):
        # 動態調整參數
        if i < split_idx:
            # Phase 1: Injection (注入期)
            # 高 LR: 強制寫入信息，確保抗攻擊
            # 低 Reg: 允許圖像偏離，優先保證信息嵌入
            current_lr = 0.25   
            current_reg = 0.5   
            phase = "Injection"
        else:
            # Phase 2: Refinement (修復期)
            # 低 LR: 微調，避免破壞已嵌入的信息
            # 高 Reg: 強制拉回原始分佈，修復 FID
            current_lr = 0.05   
            current_reg = 2.0   
            phase = "Refinement"

        z_eval = z_target if i == 0 else z_opt

        with torch.no_grad(), autocast("cuda"):
            z_0, _ = sampler.sample(steps=dpm_steps, conditioning=c, batch_size=1, shape=(4, 64, 64),
                                    unconditional_guidance_scale=scale, unconditional_conditioning=uc,
                                    x_T=z_eval, DPMencode=False, DPMdecode=True, verbose=False)
            z_rec, _ = sampler.sample(steps=dpm_steps, conditioning=c, batch_size=1, shape=(4, 64, 64),
                                      unconditional_guidance_scale=scale, unconditional_conditioning=uc,
                                      x_T=z_0, DPMencode=True, DPMdecode=False, verbose=False)
    
        diff = (z_rec - z_target).float()
        recon_loss = torch.mean(diff**2)
        reg_loss = torch.mean((z_eval - z_target)**2) if i > 0 else torch.tensor(0.0).to(device)
        
        loss = recon_loss + current_reg * reg_loss
        
        # 策略性保存最佳結果
        if i >= split_idx and loss < min_loss:
            # 只有在修復期，我們才根據 loss 來選最佳結果
            min_loss = loss
            z_best = z_eval.clone()
        elif i < split_idx:
            # 在注入期，我們總是更新，因為目標是推動 Latent 變化
            z_best = z_eval.clone()

        if i == opt_iters: break

        grad_recon = diff 
        grad_reg = 2.0 * (z_eval - z_target)
        total_gradient = grad_recon + current_reg * grad_reg
        
        guided_gradient = total_gradient * uncertainty_mask
        
        # 梯度裁剪也根據階段調整
        # 注入期允許更大的步伐，修復期步伐要小
        clip_val = 0.15 if phase == "Injection" else 0.05
        update = torch.clamp(current_lr * guided_gradient, -clip_val, clip_val)
        
        z_opt = torch.clamp(z_opt - update.to(device), -4.0, 4.0)

    # 4. Final Decode
    with torch.no_grad(), autocast("cuda"):
        z_0_final, _ = sampler.sample(steps=dpm_steps, conditioning=c, batch_size=1, shape=(4, 64, 64),
                                      unconditional_guidance_scale=scale, unconditional_conditioning=uc,
                                      x_T=z_best, DPMencode=False, DPMdecode=True, verbose=False)
        x_samples = model.decode_first_stage(z_0_final)
    
    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
    img_np = x_samples.cpu().numpy()[0].transpose(1, 2, 0) * 255
    pil_img = Image.fromarray(img_np.astype(np.uint8))
    
    # 禁用 Refinement 以獲得更好的客觀指標 (FID/BRISQUE)
    pil_img = apply_refinement(pil_img, enable=False)
    
    pil_img.save(outpath)

def run_alice():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--secret_key", type=int, required=True)
    parser.add_argument("--payload_path", type=str, required=True)
    parser.add_argument("--outpath", type=str, default="output.png")
    parser.add_argument("--init_latent", type=str, default=None)
    parser.add_argument("--opt_iters", type=int, default=15) # 稍微增加迭代次數
    parser.add_argument("--use_uncertainty", action="store_true")
    
    # 兼容性參數
    parser.add_argument("--lr", type=float, default=0.05) 
    parser.add_argument("--lambda_reg", type=float, default=1.5) 
    
    parser.add_argument("--dpm_steps", type=int, default=20)
    parser.add_argument("--scale", type=float, default=5.0)
    parser.add_argument("--ckpt", type=str, default="weights/v1-5-pruned.ckpt")
    parser.add_argument("--config", type=str, default="configs/stable-diffusion/ldm.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    opt = parser.parse_args()

    device = torch.device(opt.device)
    config = OmegaConf.load(opt.config)
    model = load_model_from_config(config, opt.ckpt, device)
    sampler = DPMSolverSampler(model)

    with open(opt.payload_path, "rb") as f: raw_data = f.read()
    payload_data = raw_data 
    CAPACITY_BYTES = 16384 // 8 
    if len(payload_data) > CAPACITY_BYTES - 2: payload_data = payload_data[:CAPACITY_BYTES-2]
    length_header = len(payload_data).to_bytes(2, 'big')
    final_payload = length_header + payload_data
    if len(final_payload) < CAPACITY_BYTES: final_payload += b'\x00' * (CAPACITY_BYTES - len(final_payload))
    
    generate_alice_image(
        model=model,
        sampler=sampler,
        prompt=opt.prompt,
        secret_key=opt.secret_key,
        payload_data=final_payload,
        outpath=opt.outpath,
        init_latent_path=opt.init_latent,
        opt_iters=opt.opt_iters,
        use_uncertainty=opt.use_uncertainty,
        dpm_steps=opt.dpm_steps,
        scale=opt.scale,
        device=opt.device
    )
    print(f"Generated: {opt.outpath}")

if __name__ == "__main__":
    try: run_alice()
    except Exception: sys.exit(1)