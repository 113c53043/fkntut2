
import torch
import numpy as np
from scipy.linalg import qr
from scipy.stats import norm
# import Levenshtein
# from S2I_Transformation_scheme2 import embed_message, extract_message


# abstract class
class mapping_module:
    def __init__(self, need_uniform_sampler=False, need_gaussian_sampler=False, bits=1, seed=None):
        self.need_uniform_sampler = need_uniform_sampler
        self.need_gaussian_sampler = need_gaussian_sampler
        self.bits = bits
        self.bits_l = 2 ** bits
        self.seed = seed
        pass

    # 将秘密信息映射为噪声, 修改式的方法往往需要借助额外的采样步骤
    def encode_secret(self, secret_message, ori_sample=None):
        pass

    # 从噪声中还原秘密信息
    def decode_secret(self, pred_noise):
        pass


# Same as TMM Mapping when bits=1
class simple_mapping(mapping_module):
    # 简单的映射方法，将秘密信息映射为正负号
    def __init__(self, bits=1):
        assert bits == 1
        super(simple_mapping, self).__init__(need_gaussian_sampler=True)

    def encode_secret(self, secret_message, ori_sample=None):
        secret_re = secret_message * 2 - 1
        ori_sign = np.sign(ori_sample)
        out = ori_sample * ori_sign * secret_re
        return out

    def decode_secret(self, pred_noise):
        out = np.round((np.sign(pred_noise) + 1) / 2)
        return out


# From paper (TMM 2023): https://ieeexplore.ieee.org/abstract/document/10306313
class tmm_mapping(mapping_module):
    def __init__(self, bits=1):
        super(tmm_mapping, self).__init__(need_uniform_sampler=True, bits=bits)

    def encode_secret(self, secret_message, ori_sample=None):
        out = norm.ppf((ori_sample + secret_message) / self.bits_l)
        return out

    def decode_secret(self, pred_noise):
        out = np.floor(self.bits_l * norm.cdf(pred_noise))
        return out

# From paper (TDSC 2022): https://ieeexplore.ieee.org/abstract/document/9931463
# class tdsc_mapping(mapping_module):
#     def __init__(self, bits, group_num=6, fixed=4096):
#         super(tdsc_mapping, self).__init__(need_gaussian_sampler=True, bits=bits)
#         self.group_num = group_num
#         self.fixed = fixed
#         self.cap = None
#
#     def encode_secret(self, secret_message, ori_sample=None, key=-1):
#         latent_shape = secret_message.shape
#         secret = ''.join([bin(int(num)).replace('0b', '').zfill(self.bits) for num in secret_message.flatten()])
#         z, c = embed_message(
#             ori_sample.flatten(), secret, self.group_num,
#             key=key, fixed_pos_list=self.fixed
#         )
#         self.cap = c
#         return z.reshape(latent_shape)
#
#     def decode_secret(self, pred_noise, key=-1):
#         out = extract_message(pred_noise.flatten(), group_num=self.group_num, key=key, fixed_pos_list=self.fixed)
#         return out
#
#     def _compute_acc(self, secret_message, received_message, length=-1):
#         secret_message = ''.join([bin(int(num)).replace('0b', '').zfill(self.bits) for num in secret_message.flatten()])
#         if length <= 0:
#             length = max(len(secret_message), len(received_message))
#         if len(secret_message) < length:
#             secret_message = secret_message + '0' * (length - len(secret_message))
#         else:
#             secret_message = secret_message[:length]
#         #     if len(str2) < length:
#         #         str2 = str2 + '0' * (length - len(str2))
#         #     else:
#         #         str2 = str2[:length]
#         return 1 - Levenshtein.distance(secret_message, received_message) / max(length, len(received_message))


class ours_mapping(mapping_module):
    def __init__(self, bits=1):
        super(ours_mapping, self).__init__(bits=bits)
        self.bits_mean = (self.bits_l - 1) / 2
        self.bits_std = ((self.bits_l ** 2 - 1) / 12) ** 0.5

    def _get_random_kernel(self, seed_kernel, kernel_shape):
        ori_seed = np.random.get_state()[1][0]  # 获取原来的随机种子
        np.random.seed(seed_kernel)
        H = np.random.randn(*kernel_shape)
        Q, r = qr(H)  # Gram-Schmidt正交化过程
        kernel = Q  #
        np.random.seed(ori_seed)  # 恢复原来的种子
        return kernel

    def _random_shuffle(self, ori_input, seed_shuffle, reverse=False):
        ori_seed = np.random.get_state()[1][0]  # 获取原来的随机种子
        np.random.seed(seed_shuffle)

        ori_shape = ori_input.shape
        ori_input = ori_input.flatten()
        ori_order = np.arange(0, len(ori_input))
        shuffle_order = ori_order.copy()
        np.random.shuffle(shuffle_order)  # 索引打乱
        if reverse:
            sorted_shuffle_order = np.argsort(shuffle_order)
            reverse_order = ori_order[sorted_shuffle_order]
            out = ori_input[reverse_order]
        else:
            out = ori_input[shuffle_order]
        out = out.reshape(*ori_shape)
        np.random.seed(ori_seed)  # 恢复原来的种子
        return out

    # 我们的映射方法 用不上额外采样
    def encode_secret(self, secret_message, ori_sample=None, seed_kernel=None, seed_shuffle=None):
        kernel = self._get_random_kernel(seed_kernel=seed_kernel, kernel_shape=secret_message.shape[-2:])  # 获取随机kernel
        secret_re = (secret_message - self.bits_mean) / self.bits_std
        out = np.matmul(np.matmul(kernel, secret_re), kernel.transpose(-1, -2))
        out = self._random_shuffle(out, seed_shuffle=seed_shuffle)  # 随机打乱
        return out

    def decode_secret(self, pred_noise, seed_kernel=None, seed_shuffle=None):
        pred_noise = self._random_shuffle(pred_noise, seed_shuffle=seed_shuffle, reverse=True)  # 取消随机打乱
        kernel = self._get_random_kernel(seed_kernel=seed_kernel, kernel_shape=pred_noise.shape[-2:])  # 获取随机kernel
        secret_hat = np.matmul(np.matmul(kernel.transpose(-1, -2), pred_noise), kernel)
        secret_hat = secret_hat * self.bits_std + self.bits_mean
        secret_hat = np.clip(secret_hat, a_min=0., a_max=float(self.bits_l - 1))
        out = np.round(secret_hat) % self.bits_l
        return out

    # ========== 【新增函數】 ==========
    # (此函數為新增，不影響 decode_secret 的原始邏輯)
    def decode_secret_soft(self, pred_noise, seed_kernel=None, seed_shuffle=None):
        """
        軟判決解碼：返回 round() 之前的浮點數值
        (Soft-Decision Decoding: Returns the floating-point values before np.round())
        """
        pred_noise = self._random_shuffle(pred_noise, seed_shuffle=seed_shuffle, reverse=True)  # 取消随机打乱
        kernel = self._get_random_kernel(seed_kernel=seed_kernel, kernel_shape=pred_noise.shape[-2:])  # 获取随机kernel
        secret_hat = np.matmul(np.matmul(kernel.transpose(-1, -2), pred_noise), kernel)
        secret_hat = secret_hat * self.bits_std + self.bits_mean
        secret_hat = np.clip(secret_hat, a_min=0., a_max=float(self.bits_l - 1))
        # 不執行 np.round()，直接返回浮點數
        return secret_hat 
    # ========== 【新增結束】 ==========


if __name__ == '__main__':
    bits = 1
    # # 我们的映射方法
    # # args = dict(seed_kernel=100, seed_shuffle=101)
    # # f = ours_mapping(bits=bits)
    #
    # # simple映射方法
    # # args = dict()
    # # f = simple_mapping()
    #
    # # tmm的映射方法
    # args = dict()
    # f = tmm_mapping(bits=bits)
    #
    # tdsc的映射方法
    
    # 由於 tdsc_mapping 相關的函數 (embed_message, extract_message) 未定義
    # 原始的 if __name__ == '__main__' 區塊會引發 NameError
    # 為了讓文件能被導入而不報錯，我們將其註解
    
    # init_args = dict(group_num=6, fixed=4096, bits=bits)
    # f = tdsc_mapping(**init_args)
    # args = dict(key=1001)
    # ori_sample = None
    # if f.need_uniform_sampler:
    #     ori_sample = np.random.rand(*(1, 4, 64, 64))
    # if f.need_gaussian_sampler:
    #     ori_sample = np.random.randn(*(1, 4, 64, 64))
    # secret = np.random.randint(0, 2**bits, (1, 4, 64, 64))  # 随机生成秘密信息
    # z = f.encode_secret(secret_message=secret, ori_sample=ori_sample, **args)
    # z_hat = z + np.random.randn(*(1, 4, 64, 64)) * 0.65
    # secret_recon = f.decode_secret(pred_noise=z_hat, **args)
    # #
    # print('bpp:', f.cap/(64*64*4))
    # print(f._compute_acc(secret, secret_recon))
    
    # ... (其餘原始的 if __name__ == '__main__' 測試代碼) ...
    pass # 添加 pass 以確保文件結構完整 
   
