import sys
import os
import argparse
import torch
import numpy as np
import traceback
from omegaconf import OmegaConf
from torch import autocast
from PIL import Image

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)
sys.path.append(os.path.join(CURRENT_DIR, "scripts"))

try:
    from mapping_module import ours_mapping
    from ldm.util import instantiate_from_config
    from ldm.models.diffusion.dpm_solver import DPMSolverSampler
except ImportError:
    sys.exit(1)

def recursive_fix_config(conf):
    if isinstance(conf, (dict, OmegaConf)):
        for key in conf.keys():
            if key == "image_size" and conf[key] == 32: conf[key] = 64
            recursive_fix_config(conf[key])

def load_model_from_config(config, ckpt, device):
    recursive_fix_config(config.model)
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

def run_alice():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--secret_key", type=int, required=True)
    parser.add_argument("--payload_path", type=str, required=True)
    parser.add_argument("--outpath", type=str, default="output.png")
    parser.add_argument("--init_latent", type=str, default=None)
    
    # 參數設定 (Pure 版只用 LR)
    parser.add_argument("--opt_iters", type=int, default=10) 
    parser.add_argument("--lr", type=float, default=0.25) 
    
    # 為了相容測試腳本的統一呼叫，這裡雖不使用但需定義，避免報錯
    parser.add_argument("--lambda_reg", type=float, default=0.0) 
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
    np.save(opt.outpath + ".gt_bits.npy", np.frombuffer(final_payload, dtype=np.uint8))

    if opt.init_latent and os.path.exists(opt.init_latent):
        z_target = torch.load(opt.init_latent, map_location=device)
    else:
        bits = np.unpackbits(np.frombuffer(final_payload, dtype=np.uint8))
        if len(bits) < 16384: bits = np.pad(bits, (0, 16384 - len(bits)), 'constant')
        bits = bits[:16384].reshape(1, 4, 64, 64)
        mapper = ours_mapping(bits=1)
        z_target_numpy = mapper.encode_secret(secret_message=bits, seed_kernel=opt.secret_key, seed_shuffle=opt.secret_key + 999)
        z_target = torch.from_numpy(z_target_numpy).float().to(device)

    # Optimization (Pure: No Mask, No Reg)
    negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
    c = model.get_learned_conditioning([opt.prompt])
    uc = model.get_learned_conditioning([negative_prompt])

    z_opt = z_target.clone()
    z_opt.requires_grad = False 
    z_best = z_target.clone() 
    min_loss = float('inf')
    current_lr = opt.lr

    if opt.opt_iters == 0:
        z_best = z_target
    else:
        for i in range(opt.opt_iters + 1):
            z_eval = z_target if i == 0 else z_opt

            with torch.no_grad(), autocast("cuda"):
                z_0, _ = sampler.sample(steps=opt.dpm_steps, conditioning=c, batch_size=1, shape=(4, 64, 64),
                                        unconditional_guidance_scale=opt.scale, unconditional_conditioning=uc,
                                        x_T=z_eval, DPMencode=False, DPMdecode=True, verbose=False)
                z_rec, _ = sampler.sample(steps=opt.dpm_steps, conditioning=c, batch_size=1, shape=(4, 64, 64),
                                          unconditional_guidance_scale=opt.scale, unconditional_conditioning=uc,
                                          x_T=z_0, DPMencode=True, DPMdecode=False, verbose=False)
        
            diff = (z_rec - z_target).float()
            loss = torch.mean(diff**2).item()
            
            if loss < min_loss:
                min_loss = loss
                z_best = z_eval.clone()
            else:
                if i > 0: current_lr *= 0.95

            if i == opt.opt_iters: break

            # Pure Gradient: Only Recon Loss
            update = torch.clamp(current_lr * diff, -0.1, 0.1)
            z_opt = torch.clamp(z_opt - update.to(device), -4.0, 4.0)

    with torch.no_grad(), autocast("cuda"):
        z_0_final, _ = sampler.sample(steps=opt.dpm_steps, conditioning=c, batch_size=1, shape=(4, 64, 64),
                                      unconditional_guidance_scale=opt.scale, unconditional_conditioning=uc,
                                      x_T=z_best, DPMencode=False, DPMdecode=True, verbose=False)
        x_samples = model.decode_first_stage(z_0_final)
    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
    Image.fromarray((x_samples.cpu().numpy()[0].transpose(1, 2, 0) * 255).astype(np.uint8)).save(opt.outpath)

if __name__ == "__main__":
    try: run_alice()
    except Exception: sys.exit(1)