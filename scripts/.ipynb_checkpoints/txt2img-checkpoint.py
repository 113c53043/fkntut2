import sys
sys.path.append('..')
import argparse, os
import torch
import numpy as np
from omegaconf import OmegaConf
from torch import autocast
from contextlib import nullcontext
from pytorch_lightning import seed_everything
from ldm.util import instantiate_from_config
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
from tqdm import tqdm
from scripts.utils import gray_code
import robust_eval
import mapping_module


def cal_acc(input, gt, gray_list, bits):
    trans_fn = np.frompyfunc(lambda x: int(gray_list[int(x)], 2), 1, 1)
    count_fn = np.frompyfunc(lambda x: bin(int(x)).count('1'), 1, 1)
    a1 = trans_fn(input).astype(np.int32)
    a2 = trans_fn(gt).astype(np.int32)
    result = a1 ^ a2
    result = count_fn(result).flatten()
    shape = len(result)
    count = sum(result)
    acc = 1 - count/(int(shape)*bits)
    return acc


def load_model_from_config(config, ckpt, gpu, verbose=False):
    if verbose:
        print(f"Loading model from {ckpt}")
        
    pl_sd = torch.load(ckpt, map_location=gpu, weights_only=False)
    if "global_step" in pl_sd and verbose:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.eval()
    return model


def load_model_and_get_prompt_embedding(model, opt, prompts):
    if opt.scale != 1.0:
        uc = model.get_learned_conditioning(opt.n_samples * [""])
    else:
        uc = None
    c = model.get_learned_conditioning(prompts)

    return c, uc


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a professional photograph of a doggy, ultra realistic",
        help="the prompt to render"
    )
    # ... (其他所有 parser.add_argument 保持不變) ...
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="./outputs"
    )
    parser.add_argument(
        "--dpm_steps",
        type=int,
        default=20,
        help="number of sampling steps",
    )
    parser.add_argument(
        "--dpm_gen_steps",
        type=int,
        default=20,
        help="number of generation steps",
    )
    parser.add_argument(
        "--dpm_inv_steps",
        type=int,
        default=20,
        help="number of inversion steps",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/stable-diffusion/v2-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="./ckpt/v2-1_512-ema-pruned.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=58756,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--dpm_order",
        type=int,
        help="",
        choices=[1, 2, 3],
        default=2
    )
    parser.add_argument(
        "--tau_a",
        type=float,
        help="",
        default=0.4
    )
    parser.add_argument(
        "--tau_b",
        type=float,
        help="",
        default=0.8
    )
    parser.add_argument(
        "--gpu",
        type=str,
        help="",
        default='cuda:0'
    )
    parser.add_argument(
        "--test_prompts",
        type=str,
        help="./test_prompts.txt",
        default='chatgpt'
    )
    parser.add_argument(
        "--attack_layer",
        type=str,
        help="which attack",
        default='gblur'
    )
    parser.add_argument(
        "--attack_factor",
        type=float,
        help="different opt",
        default=3
    )
    parser.add_argument(
        "--bit_num",
        type=int,
        help="bit nums per pixel",
        default=1
    )
    parser.add_argument(
        "--mapping_func",
        type=str,
        help="mapping_module",
        default='ours_mapping'
    )
    
    # --- 【關鍵修正】: 添加 --quiet 參數 ---
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable all non-essential print outputs (like tqdm and in-loop logs)"
    )
    # --- 【修正結束】 ---

    opt = parser.parse_args()
    device = torch.device(opt.gpu) if torch.cuda.is_available() else torch.device("cpu")
    
    if not opt.quiet:
        print(opt.dpm_steps)
    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    attack_type = opt.attack_layer
    attack_factor = opt.attack_factor
    mapping_type = opt.mapping_func
    if not opt.quiet:
        print(f'we are testing {attack_type} under setting {attack_factor}')
        
    attack_func = getattr(robust_eval, attack_type)
    base_count = len(os.listdir(outpath))
    attack_suffix = str(attack_factor).replace('.', '')
    tmp_image_name = outpath + '/tmp_%03d' % (base_count) + f'_{attack_type}_{attack_suffix}'
    bits = opt.bit_num
    gray_list = gray_code(bits)
    mapping_func = getattr(mapping_module, mapping_type)(bits=bits)

    config = OmegaConf.load(opt.config)
    model = load_model_from_config(config, opt.ckpt, opt.gpu, verbose=(not opt.quiet)) # 傳遞 verbose
    model = model.to(device)
    sampler = DPMSolverSampler(model)

    precision_scope = autocast if opt.precision == "autocast" else nullcontext

    prompt_path = opt.test_prompts
    with open(prompt_path, 'r') as f:
        prompts = f.readlines()
        prompts = [i.strip() for i in prompts]

    batch_size = opt.n_samples
    width = 512
    height = 512
    acc_records = []

    with torch.no_grad():
        with precision_scope("cuda"):
            
            # --- 【關鍵修正】: 根據 --quiet 決定是否使用 tqdm ---
            loop_iterator = range(len(prompts))
            if not opt.quiet:
                loop_iterator = tqdm(loop_iterator)

            for j in loop_iterator:
            # --- 【修正結束】 ---
            
                cur_prompt = prompts[j]
                
                if not opt.quiet:
                    print(cur_prompt)
                
                c, uc = load_model_and_get_prompt_embedding(model, opt, [cur_prompt])

                latent_shape = (batch_size, opt.C, int(height // opt.f), int(width // opt.f))

                random_input = np.random.randint(0, 2 ** bits, latent_shape)  
                random_input_ori_sample = None
                if mapping_func.need_uniform_sampler:
                    random_input_ori_sample = np.random.rand(*latent_shape)
                if mapping_func.need_gaussian_sampler:
                    random_input_ori_sample = np.random.randn(*latent_shape)
                if mapping_type == 'ours_mapping':
                    seed_shuffle = np.random.randint(0, 2 ** 32, 1)
                    seed_kernel = np.random.randint(0, 2 ** 32, 1)
                    random_input_args = dict(seed_kernel=seed_kernel, seed_shuffle=seed_shuffle)
                elif mapping_type == 'tdsc_mapping':
                    random_input_args = dict(key=np.random.randint(0, 2 ** 32, 1))
                else:
                    random_input_args = dict()
                init_latent = mapping_func.encode_secret(secret_message=random_input, ori_sample=random_input_ori_sample, **random_input_args).astype(np.float32)
                init_latent = torch.from_numpy(init_latent).to(device)

                shape = init_latent.shape[1:]
                z_0, _ = sampler.sample(steps=opt.dpm_gen_steps,
                                        unconditional_conditioning=uc,
                                        conditioning=c,
                                        batch_size=opt.n_samples,
                                        shape=shape,
                                        verbose=False,
                                        unconditional_guidance_scale=opt.scale,
                                        eta=opt.ddim_eta,
                                        order=opt.dpm_order,
                                        x_T=init_latent,
                                        width=width,
                                        height=height,
                                        DPMencode=False,
                                        DPMdecode=True,
                                        )

                x0_samples = model.decode_first_stage(z_0)

                x0_samples = attack_func(x0_samples, factor=attack_factor, tmp_image_name=tmp_image_name).to(device)

                init_latent_hat = model.get_first_stage_encoding(model.encode_first_stage(x0_samples))
                z_enc, _ = sampler.sample(steps=opt.dpm_inv_steps,
                                          unconditional_conditioning=uc,
                                          conditioning=c,
                                          batch_size=opt.n_samples,
                                          shape=shape,
                                          verbose=False,
                                          unconditional_guidance_scale=opt.scale,
                                          eta=opt.ddim_eta,
                                          order=opt.dpm_order,
                                          x_T=init_latent_hat,
                                          width=width,
                                          height=height,
                                          DPMencode=True,
                                          )
                
                if not opt.quiet:
                    distance = (init_latent - init_latent_hat).abs().mean()
                    print("encoder-decode error:", distance)
                    recon_distance = (init_latent - z_enc).abs().mean()
                    print("recon-error", recon_distance)
                
                pred_noise = z_enc.clone().cpu().numpy()
                recon_latent = mapping_func.decode_secret(pred_noise=pred_noise, **random_input_args)
                if mapping_type == 'tdsc_mapping':
                    acc = mapping_func._compute_acc(random_input, recon_latent)
                else:
                    acc = cal_acc(recon_latent, random_input, gray_list=gray_list, bits=bits)
                
                if not opt.quiet:
                    print(f'mean:{acc}')
                acc_records.append(acc)

    # 最終的 print 必須保留
    print(f'average accuracy: {sum(acc_records) / len(acc_records)}')

if __name__ == "__main__":
    main()