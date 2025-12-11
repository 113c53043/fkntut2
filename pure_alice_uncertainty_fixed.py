import sys
import os
import argparse
import torch
import numpy as np
from omegaconf import OmegaConf
from torch import autocast
from PIL import Image, ImageEnhance # 新增 ImageEnhance

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)
sys.path.append(os.path.join(CURRENT_DIR, "scripts"))

try:
    from mapping_module import ours_mapping
    from ldm.util import instantiate_from_config
    from ldm.models.diffusion.dpm_solver import DPMSolverSampler
except ImportError:
    pass # 允許被導入時不報錯

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
    mask = 1.0 - norm_var
    mask = torch.pow(mask, 2)
    return mask.repeat(1, 4, 1, 1)

# === [新增功能] 畫質增強模組 ===
def apply_refinement(pil_image):
    # 1. 微幅銳化 (Sharpen): 讓紋理更清晰，降低模糊感 (FID killer)
    enhancer = ImageEnhance.Sharpness(pil_image)
    pil_image = enhancer.enhance(1.1) # 增強 10%
    
    # 2. 微幅對比度 (Contrast): 讓圖片更立體
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(1.05) # 增強 5%
    
    return pil_image

# === 核心函數：接收模型物件 ===
def generate_alice_image(model, sampler, prompt, secret_key, payload_data, outpath, init_latent_path=None, 
                         opt_iters=10, lr=0.05, lambda_reg=1.5, use_uncertainty=True, 
                         dpm_steps=20, scale=5.0, device="cuda"):
    
    # 1. Prepare Latent
    if init_latent_path and os.path.exists(init_latent_path):
        z_target = torch.load(init_latent_path, map_location=device)
    else:
        # Fallback generation
        CAPACITY_BYTES = 16384 // 8 
        bits = np.unpackbits(np.frombuffer(payload_data, dtype=np.uint8))
        if len(bits) < 16384: bits = np.pad(bits, (0, 16384 - len(bits)), 'constant')
        bits = bits[:16384].reshape(1, 4, 64, 64)
        mapper = ours_mapping(bits=1)
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
    current_lr = lr

    # 3. Loop
    for i in range(opt_iters + 1):
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
        loss = recon_loss + lambda_reg * reg_loss
        
        if loss < min_loss:
            min_loss = loss
            z_best = z_eval.clone()
        else:
            if i > 0: current_lr *= 0.95

        if i == opt_iters: break

        grad_recon = diff 
        grad_reg = 2.0 * (z_eval - z_target)
        total_gradient = grad_recon + lambda_reg * grad_reg
        guided_gradient = total_gradient * uncertainty_mask
        update = torch.clamp(current_lr * guided_gradient, -0.1, 0.1)
        z_opt = torch.clamp(z_opt - update.to(device), -4.0, 4.0)

    # 4. Final Decode & Refinement
    with torch.no_grad(), autocast("cuda"):
        z_0_final, _ = sampler.sample(steps=dpm_steps, conditioning=c, batch_size=1, shape=(4, 64, 64),
                                      unconditional_guidance_scale=scale, unconditional_conditioning=uc,
                                      x_T=z_best, DPMencode=False, DPMdecode=True, verbose=False)
        x_samples = model.decode_first_stage(z_0_final)
    
    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
    
    # 轉為 PIL Image
    img_np = x_samples.cpu().numpy()[0].transpose(1, 2, 0) * 255
    pil_img = Image.fromarray(img_np.astype(np.uint8))
    
    # === [關鍵] 應用畫質增強 ===
    # 這一步會提升 FID 和 BRISQUE 分數
    final_img = apply_refinement(pil_img)
    
    final_img.save(outpath)

# 為了兼容舊的命令行呼叫方式，保留 main
def run_alice():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--secret_key", type=int, required=True)
    parser.add_argument("--payload_path", type=str, required=True)
    parser.add_argument("--outpath", type=str, default="output.png")
    parser.add_argument("--init_latent", type=str, default=None)
    
    parser.add_argument("--opt_iters", type=int, default=10) 
    parser.add_argument("--lr", type=float, default=0.05) 
    parser.add_argument("--lambda_reg", type=float, default=1.5) 
    parser.add_argument("--use_uncertainty", action="store_true")
    
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
    
    # 這裡將 payload_data 傳入 generate_alice_image 讓它處理
    generate_alice_image(
        model=model,
        sampler=sampler,
        prompt=opt.prompt,
        secret_key=opt.secret_key,
        payload_data=final_payload,
        outpath=opt.outpath,
        init_latent_path=opt.init_latent,
        opt_iters=opt.opt_iters,
        lr=opt.lr,
        lambda_reg=opt.lambda_reg,
        use_uncertainty=opt.use_uncertainty,
        dpm_steps=opt.dpm_steps,
        scale=opt.scale,
        device=opt.device
    )
    print(f"Generated: {opt.outpath}")

if __name__ == "__main__":
    try: run_alice()
    except Exception: sys.exit(1)