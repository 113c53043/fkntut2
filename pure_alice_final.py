import os
import sys

# === 1. 路徑設定 (必須最先執行) ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)
# 加入 scripts 資料夾以防萬一
SCRIPTS_DIR = os.path.join(MAS_GRDH_PATH, "scripts") if 'MAS_GRDH_PATH' in locals() else os.path.join(PARENT_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.append(SCRIPTS_DIR)

import argparse
import torch
import numpy as np
from omegaconf import OmegaConf
from torch import autocast
from PIL import Image, ImageEnhance

try:
    from mapping_module import ours_mapping
    from ldm.util import instantiate_from_config
    from ldm.models.diffusion.dpm_solver import DPMSolverSampler
except ImportError as e:
    print(f"❌ Import Error in Alice: {e}")
    sys.exit(1)

# === 統一的 Negative Prompt ===
LONG_NEGATIVE_PROMPT = "worst quality, low quality, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, normal quality, jpeg artifacts, signature, watermark, username, blurry, bad feet, extra arms, extra legs, extra body, poorly drawn hands, missing arms, missing legs, extra hands, mangled fingers, extra fingers, disconnected limbs, mutated hands, long neck, duplicate, bad composition, malformed limbs, deformed, mutated, ugly, disgusting, amputation, cartoon, anime, 3d, illustration, talking, two bodies, double torso, three arms, three legs, bad framing, mutated face, deformed face, cross-eyed, body out of frame, cloned face, disfigured, fused fingers, too many fingers, long fingers, gross proportions, poorly drawn face, text focus, bad focus, out of focus, extra nipples, missing nipples, fused nipples, extra breasts, enlarged breasts, deformed breasts, bad shadow, overexposed, underexposed, bad lighting, color distortion, weird colors, dull colors, bad eyes, dead eyes, asymmetrical eyes, hollow eyes, collapsed eyes, mutated eyes, distorted iris, wrong eye position, wrong teeth, crooked teeth, melted teeth, distorted mouth, wrong lips, mutated lips, broken lips, twisted mouth, bad hair, coarse hair, messy hair, artifact hair, unnatural hair texture, missing hair, polygon hair, bad skin, oily skin, plastic skin, uneven skin, dirty skin, pores, face holes, oversharpen, overprocessed, nsfw, extra tongue, long tongue, split tongue, bad tongue, distorted tongue, blurry background, messy background, multiple heads, split head, fused head, broken head, missing head, duplicated head, wrong head, loli, child, kid, underage, boy, girl, infant, toddler, baby, baby face, young child, teen, 3D render, extra limb, twisted limb, broken limb, warped limb, oversized limb, undersized limb, smudge, glitch, errors, canvas frame, cropped head, cropped face, cropped body, depth-of-field error, weird depth, lens distortion, chromatic aberration, duplicate face, wrong face, face mismatch, hands behind back, incorrect fingers, extra joint, broken joint, doll-like, mannequin, porcelain skin, waxy skin, clay texture, incorrect grip, wrong pose, unnatural pose, floating object, floating limbs, floating head, missing shadow, unnatural shadow, dislocated shoulder, bad cloth, cloth error, clothing glitch, unnatural clothing folds, stretched fabric, corrupted texture, mosaic, censored, body distortion, bent spine, malformed spine, unnatural spine angle, twisted waist, extra waist, glowing eyes, horror eyes, scary face, mutilated, blood, gore, wounds, injury, amputee, long body, short body, bad perspective, impossible perspective, broken perspective, wrong angle, disfigured eyes, lazy eye, cyclops, extra eye, mutated body, malformed body, clay skin, huge head, tiny head, uneven head, incorrect anatomy, missing torso, half torso, torso distortion"

def load_model_from_config(config, ckpt, device):
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

def estimate_uncertainty(model, sampler, z_center, c, uc, scale, device, repeats=4, noise_std=0.05, mode="adaptive"):
    if mode == "pure":
        return torch.ones_like(z_center)

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
    
    if mode == "fixed":
        # === Fixed Strategy ===
        v_min = variance_mean.min()
        v_max = variance_mean.max()
        norm_var = (variance_mean - v_min) / (v_max - v_min + 1e-8)
        mask = 1.0 - norm_var
        mask = torch.pow(mask, 2)
        mask = mask * 0.7 + 0.3 
        
    elif mode == "adaptive":
        # === Optimal Adaptive Strategy (Power 6, Floor 0.4) ===
        v_min = torch.quantile(variance_mean, 0.01) 
        v_max = torch.quantile(variance_mean, 0.99)
        denom = v_max - v_min
        if denom < 1e-8: denom = 1.0
        norm_var = (variance_mean - v_min) / denom
        norm_var = torch.clamp(norm_var, 0.0, 1.0)

        norm_var_powered = torch.pow(norm_var, 6.0) 
        mask = 1.0 - norm_var_powered
        
        avg_uncertainty = torch.mean(norm_var).item()
        base_floor = 0.40 + (0.3 * avg_uncertainty)
        base_floor = min(max(base_floor, 0.40), 0.70)
        
        mask = mask * (1.0 - base_floor) + base_floor

    return mask.repeat(1, 4, 1, 1)

def apply_refinement(pil_image):
    enhancer = ImageEnhance.Sharpness(pil_image)
    pil_image = enhancer.enhance(1.05) 
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(1.02) 
    return pil_image

def generate_alice_image(model, sampler, prompt, secret_key, payload_data, outpath, init_latent_path=None, 
                         opt_iters=10, lr=0.05, lambda_reg=1.5, mode="adaptive", 
                         dpm_steps=20, scale=5.0, device="cuda"):
    
    if init_latent_path and os.path.exists(init_latent_path):
        z_target = torch.load(init_latent_path, map_location=device)
    else:
        CAPACITY_BYTES = 16384 // 8 
        bits = np.unpackbits(np.frombuffer(payload_data, dtype=np.uint8))
        if len(bits) < 16384: bits = np.pad(bits, (0, 16384 - len(bits)), 'constant')
        bits = bits[:16384].reshape(1, 4, 64, 64)
        mapper = ours_mapping(bits=1)
        z_target_numpy = mapper.encode_secret(secret_message=bits, seed_kernel=secret_key, seed_shuffle=secret_key + 999)
        z_target = torch.from_numpy(z_target_numpy).float().to(device)

    if mode == "baseline":
        opt_iters = 0 
    
    c = model.get_learned_conditioning([prompt])
    uc = model.get_learned_conditioning([LONG_NEGATIVE_PROMPT])

    if opt_iters > 0:
        uncertainty_mask = estimate_uncertainty(model, sampler, z_target, c, uc, scale, device, mode=mode)
    else:
        uncertainty_mask = None 

    z_opt = z_target.clone()
    z_opt.requires_grad = False 
    z_best = z_target.clone() 
    min_loss = float('inf')
    initial_lr = lr

    for i in range(opt_iters + 1):
        if opt_iters == 0: break 

        progress = i / (opt_iters + 1)
        decay_factor = 1.0 - (0.5 * progress) 
        current_lr = initial_lr * decay_factor

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
        
        if i == opt_iters: break

        grad_recon = diff 
        grad_reg = 2.0 * (z_eval - z_target)
        total_gradient = grad_recon + lambda_reg * grad_reg
        guided_gradient = total_gradient * uncertainty_mask
        
        update = torch.clamp(current_lr * guided_gradient, -0.1, 0.1)
        z_opt = torch.clamp(z_opt - update.to(device), -4.0, 4.0)
    
    if mode == "baseline":
        z_best = z_target

    with torch.no_grad(), autocast("cuda"):
        z_0_final, _ = sampler.sample(steps=dpm_steps, conditioning=c, batch_size=1, shape=(4, 64, 64),
                                      unconditional_guidance_scale=scale, unconditional_conditioning=uc,
                                      x_T=z_best, DPMencode=False, DPMdecode=True, verbose=False)
        x_samples = model.decode_first_stage(z_0_final)
    
    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
    
    img_np = x_samples.cpu().numpy()[0].transpose(1, 2, 0) * 255
    pil_img = Image.fromarray(img_np.astype(np.uint8))
    final_img = apply_refinement(pil_img)
    final_img.save(outpath)

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
    parser.add_argument("--mode", type=str, default="adaptive", choices=["baseline", "pure", "fixed", "adaptive"])
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
        model=model, sampler=sampler, prompt=opt.prompt, secret_key=opt.secret_key,
        payload_data=final_payload, outpath=opt.outpath, init_latent_path=opt.init_latent,
        opt_iters=opt.opt_iters, lr=opt.lr, lambda_reg=opt.lambda_reg,
        mode=opt.mode, dpm_steps=opt.dpm_steps, scale=opt.scale, device=opt.device
    )

if __name__ == "__main__":
    try: run_alice()
    except Exception: sys.exit(1)