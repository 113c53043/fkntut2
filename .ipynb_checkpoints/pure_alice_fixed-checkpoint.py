import sys
import os
import argparse
import torch
import numpy as np
import traceback
from omegaconf import OmegaConf
from torch import autocast
from PIL import Image

# === Path Setup ===
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)
sys.path.append(os.path.join(CURRENT_DIR, "scripts"))

try:
    from mapping_module import ours_mapping
except ImportError:
    print("❌ Critical: mapping_module not found. Check PYTHONPATH.")
    sys.exit(1)

try:
    from ldm.util import instantiate_from_config
    from ldm.models.diffusion.dpm_solver import DPMSolverSampler
except ImportError:
    print("❌ Critical: ldm module not found. Check PYTHONPATH.")
    sys.exit(1)

def recursive_fix_config(conf):
    if isinstance(conf, (dict, OmegaConf)):
        for key in conf.keys():
            if key == "image_size" and conf[key] == 32:
                conf[key] = 64
            recursive_fix_config(conf[key])

def load_model_from_config(config, ckpt, device):
    recursive_fix_config(config.model)
    print(f"Loading model from {ckpt}...")
    try:
        pl_sd = torch.load(ckpt, map_location="cpu", weights_only=False) 
    except TypeError:
        pl_sd = torch.load(ckpt, map_location="cpu")

    if "state_dict" in pl_sd:
        sd = pl_sd["state_dict"]
    else:
        sd = pl_sd
        
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    return model

def main():
    try:
        run_alice()
    except Exception:
        print("\n❌ Alice CRASHED with the following error:")
        traceback.print_exc()
        sys.exit(1)

def run_alice():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--secret_key", type=int, required=True)
    parser.add_argument("--payload_path", type=str, required=True)
    parser.add_argument("--outpath", type=str, default="output.png")
    
    # === 新增參數：外部 Latent 路徑 ===
    parser.add_argument("--init_latent", type=str, default=None, help="Path to pre-calculated z_target .pt file")
    
    # 優化參數
    parser.add_argument("--opt_iters", type=int, default=10) 
    parser.add_argument("--lr", type=float, default=0.25) 
    parser.add_argument("--noise_std", type=float, default=0.0)
    
    parser.add_argument("--ckpt", type=str, default="weights/v1-5-pruned.ckpt")
    parser.add_argument("--config", type=str, default="configs/stable-diffusion/ldm.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dpm_steps", type=int, default=20)
    parser.add_argument("--scale", type=float, default=5.0)
    opt = parser.parse_args()

    device = torch.device(opt.device)
    config = OmegaConf.load(opt.config)
    model = load_model_from_config(config, opt.ckpt, device)
    sampler = DPMSolverSampler(model)

    # 1. Payload Preparation (仍需執行以確保 gt_bits 產生)
    with open(opt.payload_path, "rb") as f: raw_data = f.read()
    payload_data = raw_data 

    CAPACITY_BYTES = 16384 // 8 
    if len(payload_data) > CAPACITY_BYTES - 2:
        payload_data = payload_data[:CAPACITY_BYTES-2]
    
    length_header = len(payload_data).to_bytes(2, 'big')
    final_payload = length_header + payload_data
    if len(final_payload) < CAPACITY_BYTES:
        final_payload += b'\x00' * (CAPACITY_BYTES - len(final_payload))
        
    # print(f"[Info] Payload Size: {len(final_payload)} bytes")
    gt_bits_path = opt.outpath + ".gt_bits.npy"
    np.save(gt_bits_path, np.frombuffer(final_payload, dtype=np.uint8))

    # 2. Initialization (Latent Loading Logic)
    if opt.init_latent and os.path.exists(opt.init_latent):
        print(f"📥 [Init] Loading pre-calculated latent from {opt.init_latent}...")
        z_target = torch.load(opt.init_latent, map_location=device)
    else:
        # Fallback to original generation logic
        print("⚙️  [Init] Generating Target Latent via Orthogonal Mapping...")
        bits = np.unpackbits(np.frombuffer(final_payload, dtype=np.uint8))
        if len(bits) < 16384:
            bits = np.pad(bits, (0, 16384 - len(bits)), 'constant')
        bits = bits[:16384].reshape(1, 4, 64, 64)
        
        mapper = ours_mapping(bits=1)
        z_target_numpy = mapper.encode_secret(
            secret_message=bits, 
            seed_kernel=opt.secret_key, 
            seed_shuffle=opt.secret_key + 999
        )
        z_target = torch.from_numpy(z_target_numpy).float().to(device)

    # 3. Best-Checkpoint Optimization Loop
    z_opt = z_target.clone()
    z_opt.requires_grad = False 
    
    z_best = None
    min_loss = float('inf')
    best_iter = -1

    negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
    
    c = model.get_learned_conditioning([opt.prompt])
    uc = model.get_learned_conditioning([negative_prompt])
    
    # print(f"✅ Starting Optimization (Max Iters={opt.opt_iters})...")
    
    current_lr = opt.lr
    current_noise = opt.noise_std

    for i in range(opt.opt_iters + 1):
        if i == 0:
            z_eval = z_target
            prefix = "Base  "
        else:
            z_eval = z_opt
            prefix = f"Iter {i:<2}"

        with torch.no_grad(), autocast("cuda"):
            # A. Forward Generation
            z_0, _ = sampler.sample(steps=opt.dpm_steps, conditioning=c, batch_size=1, shape=(4, 64, 64),
                                    unconditional_guidance_scale=opt.scale, unconditional_conditioning=uc,
                                    x_T=z_eval, DPMencode=False, DPMdecode=True, verbose=False)
            
            # B. Simulated Noise
            if current_noise > 0 and i > 0:
                noise = torch.randn_like(z_0) * current_noise
                z_0_input = z_0 + noise
            else:
                z_0_input = z_0

            # C. Inversion
            z_rec, _ = sampler.sample(steps=opt.dpm_steps, conditioning=c, batch_size=1, shape=(4, 64, 64),
                                      unconditional_guidance_scale=opt.scale, unconditional_conditioning=uc,
                                      x_T=z_0_input, DPMencode=True, DPMdecode=False, verbose=False)
       
        # D. Loss Calculation
        diff = (z_rec - z_target).float()
        loss = torch.mean(diff**2).item()
        
        # E. Update Best Model
        is_best = False
        if loss < min_loss:
            min_loss = loss
            z_best = z_eval.clone()
            best_iter = i
            is_best = True
            improved_msg = "✅"
        else:
            improved_msg = ""
            if i > 0: current_lr *= 0.98

        # print(f"  {prefix} | Loss: {loss:.6f} {improved_msg}")

        if loss < 1e-6: break
        if i == opt.opt_iters: break

        # F. Gradient Update
        update = torch.clamp(current_lr * diff, -0.1, 0.1)
        z_opt = torch.clamp(z_opt - update.to(device), -4.0, 4.0)

    # 4. Final Generation
    # print(f"🏆 Final Selection: Iter {best_iter} with Loss {min_loss:.6f}")
    
    with torch.no_grad(), autocast("cuda"):
        z_0_final, _ = sampler.sample(steps=opt.dpm_steps, conditioning=c, batch_size=1, shape=(4, 64, 64),
                                      unconditional_guidance_scale=opt.scale, unconditional_conditioning=uc,
                                      x_T=z_best, DPMencode=False, DPMdecode=True, verbose=False)
        x_samples = model.decode_first_stage(z_0_final)
        
    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
    
    if not np.isnan(x_samples.cpu().numpy()).any():
        Image.fromarray((x_samples.cpu().numpy()[0].transpose(1, 2, 0) * 255).astype(np.uint8)).save(opt.outpath)
        print(f"✅ [Alice Pure] Generated Stego Image: {opt.outpath}")
    else:
        print("❌ Error: Generated image contains NaN")

if __name__ == "__main__":
    main()