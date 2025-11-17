"""
This file is writing for testing robustness of our method
(這是「攻擊工具庫」，由 robust_eval_main.py 調用)
"""
from utils import image_grid, load_512
import cv2
import torch
import numpy as np


def _store(samples, tmp_image_name):
    # 確保 samples 是 CPU 上的 tensor
    if isinstance(samples, torch.Tensor):
        samples = samples.cpu()
        
    image_grid(samples).save(f'{tmp_image_name}.png')
    # x0_samples = load_512('./tmp.png')
    samples = cv2.imread(f'{tmp_image_name}.png')
    samples = cv2.cvtColor(samples, cv2.COLOR_BGR2RGB)
    return samples


def _store_jpeg(samples, factor, tmp_image_name):
    # 確保 samples 是 CPU 上的 tensor
    if isinstance(samples, torch.Tensor):
        samples = samples.cpu()

    image_grid(samples).save(f'{tmp_image_name}.jpg', quality=factor)
    # x0_samples = load_512('./tmp.png')
    samples = cv2.imread(f'{tmp_image_name}.jpg')
    samples = cv2.cvtColor(samples, cv2.COLOR_BGR2RGB)
    return samples


# without any lossy operations
def identity(samples, factor=None, tmp_image_name='tmp'):
    # 為了與其他攻擊函數保持一致，我們也執行一次存儲和加載
    samples = _store(samples, tmp_image_name=tmp_image_name)
    reload_samples = load_512(samples)
    return reload_samples


# saving and reloading
def storage(samples, factor=None, tmp_image_name='tmp'):
    samples = _store(samples, tmp_image_name=tmp_image_name)
    reload_samples = load_512(samples)
    return reload_samples


def resize(samples, factor, tmp_image_name='tmp'):
    samples = _store(samples, tmp_image_name=tmp_image_name)
    height, width = samples.shape[:2]
    new_height = int(height * factor)
    new_width = int(width * factor)
    scaled_samples = cv2.resize(samples, (new_width, new_height), interpolation=cv2.INTER_LINEAR)  # 使用双线性插值
    recover_samples = cv2.resize(scaled_samples, (width, height), interpolation=cv2.INTER_LINEAR)  # 使用双线性插值
    
    # 【修正】resize 後需要保存，以便 Bob 讀取
    cv2.imwrite(f'{tmp_image_name}.png', cv2.cvtColor(recover_samples, cv2.COLOR_RGB2BGR))
    
    reload_samples = load_512(recover_samples)
    return reload_samples


def jpeg(samples, factor, tmp_image_name='tmp'):
    samples = _store_jpeg(samples, int(factor), tmp_image_name=tmp_image_name)
    reload_samples = load_512(samples)
    return reload_samples


def mblur(samples, factor, tmp_image_name='tmp'):
    samples = _store(samples, tmp_image_name=tmp_image_name)
    
    # 【修正】核心大小必須是正奇數
    ksize = int(factor)
    if ksize % 2 == 0:
        ksize += 1
        
    samples = cv2.medianBlur(samples, ksize)
    
    # 【修正】模糊後需要保存，以便 Bob 讀取
    cv2.imwrite(f'{tmp_image_name}.png', cv2.cvtColor(samples, cv2.COLOR_RGB2BGR))
    
    reload_samples = load_512(samples)
    return reload_samples


def gblur(samples, factor, tmp_image_name='tmp'):
    samples = _store(samples, tmp_image_name=tmp_image_name)
    
    # 【修正】核心大小必須是正奇數
    ksize = int(factor)
    if ksize % 2 == 0: # <-- 【關鍵修正】: 將 'O' 改為 '0'
        ksize += 1
        
    samples = cv2.GaussianBlur(samples, (ksize, ksize), 0)
    
    # 【修正】模糊後需要保存，以便 Bob 讀取
    cv2.imwrite(f'{tmp_image_name}.png', cv2.cvtColor(samples, cv2.COLOR_RGB2BGR))

    reload_samples = load_512(samples)
    return reload_samples


def awgn(samples, factor, tmp_image_name='tmp'):
    # 注意：awgn (高斯噪聲) 與其他攻擊不同，它在 Tensor 空間中操作
    
    # 確保 samples 是 Tensor
    if not isinstance(samples, torch.Tensor):
        samples = load_512(samples)
    
    # 確保
    if 'cpu' in samples.device.type:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        samples = samples.to(device)

    noise = torch.tensor(np.random.normal(0, factor, samples.shape)).to(samples.device, dtype=samples.dtype)
    samples = samples + noise  # 高斯噪声
    reload_samples = torch.clamp(samples, min=-1., max=1.)
    
    # 【修正】添加噪聲後需要保存，以便 Bob 讀取
    # 需要將 Tensor 轉回 PIL Image 以使用 image_grid
    
    # 簡易轉換 (假設 batch size 為 1)
    img_tensor = reload_samples.squeeze(0).cpu() # [C, H, W]
    
    # 將 [-1, 1] 轉為 [0, 1]
    img_tensor = (img_tensor + 1.0) / 2.0
    img_tensor = torch.clamp(img_tensor, min=0.0, max=1.0)
    
    from torchvision.transforms import ToPILImage
    pil_img = ToPILImage()(img_tensor)
    pil_img.save(f'{tmp_image_name}.png')

    return reload_samples