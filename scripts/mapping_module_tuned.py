import numpy as np
from scipy.linalg import qr
from scipy.stats import norm

# abstract class
class mapping_module:
    def __init__(self, need_uniform_sampler=False, need_gaussian_sampler=False, bits=1, seed=None):
        self.need_uniform_sampler = need_uniform_sampler
        self.need_gaussian_sampler = need_gaussian_sampler
        self.bits = bits
        self.bits_l = 2 ** bits
        self.seed = seed
        pass

    def encode_secret(self, secret_message, ori_sample=None):
        pass

    def decode_secret(self, pred_noise):
        pass

# ... (Simple and TMM mapping classes omitted for brevity, keep them if needed) ...

class ours_mapping_tuned(mapping_module):
    """
    Fine-tuned version of the original ours_mapping.
    Adds 'scale' for signal strength control and 'clipping' for artifact suppression.
    """
    def __init__(self, bits=1, scale=1.0, clip_range=3.5):
        super(ours_mapping_tuned, self).__init__(bits=bits)
        self.bits_mean = (self.bits_l - 1) / 2
        self.bits_std = ((self.bits_l ** 2 - 1) / 12) ** 0.5
        
        # === 微調參數 ===
        self.scale = scale          # 增益係數：<1.0 提升 FID, >1.0 提升魯棒性
        self.clip_range = clip_range # 截斷範圍：防止極端值導致 VAE 產生偽影 (Artifacts)

    def _get_random_kernel(self, seed_kernel, kernel_shape):
        ori_seed = np.random.get_state()[1][0]
        np.random.seed(seed_kernel)
        H = np.random.randn(*kernel_shape)
        Q, r = qr(H)
        kernel = Q
        np.random.seed(ori_seed)
        return kernel

    def _random_shuffle(self, ori_input, seed_shuffle, reverse=False):
        ori_seed = np.random.get_state()[1][0]
        np.random.seed(seed_shuffle)

        ori_shape = ori_input.shape
        ori_input = ori_input.flatten()
        ori_order = np.arange(0, len(ori_input))
        shuffle_order = ori_order.copy()
        np.random.shuffle(shuffle_order)
        if reverse:
            sorted_shuffle_order = np.argsort(shuffle_order)
            reverse_order = ori_order[sorted_shuffle_order]
            out = ori_input[reverse_order]
        else:
            out = ori_input[shuffle_order]
        out = out.reshape(*ori_shape)
        np.random.seed(ori_seed)
        return out

    def encode_secret(self, secret_message, ori_sample=None, seed_kernel=None, seed_shuffle=None):
        # 1. Generate Kernel
        kernel = self._get_random_kernel(seed_kernel=seed_kernel, kernel_shape=secret_message.shape[-2:])
        
        # 2. Normalize Message
        secret_re = (secret_message - self.bits_mean) / self.bits_std
        
        # 3. Orthogonal Transformation (Z = C * M * C^T)
        out = np.matmul(np.matmul(kernel, secret_re), kernel.transpose(-1, -2))
        
        # 4. Shuffle
        out = self._random_shuffle(out, seed_shuffle=seed_shuffle)
        
        # === 微調重點 1: 增益控制 (Scaling) ===
        # 如果 scale=0.9，噪聲能量降低，對 FID 有幫助
        out = out * self.scale
        
        # === 微調重點 2: 安全截斷 (Safety Clipping) ===
        # 強制將數值限制在 [-3.5, 3.5]
        # VAE 對 >4.0 的數值非常敏感，會產生奇怪的光斑
        if self.clip_range is not None:
            out = np.clip(out, -self.clip_range, self.clip_range)
            
        return out

    def decode_secret(self, pred_noise, seed_kernel=None, seed_shuffle=None):
        # 解碼時需要考慮 scale 的影響，特別是對於軟判決
        # 對於硬判決 (round)，scale 的影響較小，但為了數學嚴謹性應該還原
        
        # 1. Reverse Scale
        pred_noise = pred_noise / self.scale
        
        # 2. Unshuffle
        pred_noise = self._random_shuffle(pred_noise, seed_shuffle=seed_shuffle, reverse=True)
        
        # 3. Inverse Transform
        kernel = self._get_random_kernel(seed_kernel=seed_kernel, kernel_shape=pred_noise.shape[-2:])
        secret_hat = np.matmul(np.matmul(kernel.transpose(-1, -2), pred_noise), kernel)
        
        # 4. De-normalize
        secret_hat = secret_hat * self.bits_std + self.bits_mean
        secret_hat = np.clip(secret_hat, a_min=0., a_max=float(self.bits_l - 1))
        
        # 5. Rounding
        out = np.round(secret_hat) % self.bits_l
        return out

    def decode_secret_soft(self, pred_noise, seed_kernel=None, seed_shuffle=None):
        # 用於 Bob 端的軟解碼 (浮點數)
        pred_noise = pred_noise / self.scale
        pred_noise = self._random_shuffle(pred_noise, seed_shuffle=seed_shuffle, reverse=True)
        kernel = self._get_random_kernel(seed_kernel=seed_kernel, kernel_shape=pred_noise.shape[-2:])
        secret_hat = np.matmul(np.matmul(kernel.transpose(-1, -2), pred_noise), kernel)
        secret_hat = secret_hat * self.bits_std + self.bits_mean
        secret_hat = np.clip(secret_hat, a_min=0., a_max=float(self.bits_l - 1))
        return secret_hat