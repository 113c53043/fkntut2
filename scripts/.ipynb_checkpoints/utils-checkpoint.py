import PIL
from PIL import Image
from matplotlib import pyplot as plt
import torchvision.transforms as T
import torch
import numpy as np


def show_torch_img(img):
    img = to_np_image(img)
    plt.imshow(img)
    plt.axis("off")


def to_np_image(all_images):
    all_images = (all_images.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()[0]
    return all_images


def tensor_to_pil(tensor_imgs):
    if type(tensor_imgs) == list:
        tensor_imgs = torch.cat(tensor_imgs)
    tensor_imgs = (tensor_imgs / 2 + 0.5).clamp(0, 1)
    to_pil = T.ToPILImage()
    pil_imgs = [to_pil(img) for img in tensor_imgs]
    return pil_imgs


def pil_to_tensor(pil_imgs):
    to_torch = T.ToTensor()
    if type(pil_imgs) == PIL.Image.Image:
        tensor_imgs = to_torch(pil_imgs).unsqueeze(0) * 2 - 1
    elif type(pil_imgs) == list:
        tensor_imgs = torch.cat([to_torch(pil_imgs).unsqueeze(0) * 2 - 1 for img in pil_imgs]).to(device)
    else:
        raise Exception("Input need to be PIL.Image or list of PIL.Image")
    return tensor_imgs


def add_margin(pil_img, top=0, right=0, bottom=0,
               left=0, color=(255, 255, 255)):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)

    result.paste(pil_img, (left, top))
    return result


def image_grid(imgs, rows=1, cols=None,
               size=None):
    if type(imgs) == list and type(imgs[0]) == torch.Tensor:
        imgs = torch.cat(imgs)
    if type(imgs) == torch.Tensor:
        imgs = tensor_to_pil(imgs)

    if not size is None:
        imgs = [img.resize((size, size)) for img in imgs]
    if cols is None:
        cols = len(imgs)
    assert len(imgs) >= rows * cols

    top = 20
    w, h = imgs[0].size
    delta = 0
    if len(imgs) > 1 and not imgs[1].size[1] == h:
        delta = top
        h = imgs[1].size[1]
    grid = Image.new('RGB', size=(cols * w, rows * h + delta))
    for i, img in enumerate(imgs):
        if not delta == 0 and i > 0:
            grid.paste(img, box=(i % cols * w, i // cols * h + delta))
        else:
            grid.paste(img, box=(i % cols * w, i // cols * h))

    return grid


# 读取一张图像 返回值大小 [1 3 512 512]
def load_512(image_path, left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        image = np.array(Image.open(image_path).convert('RGB'))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w-1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h-bottom, left:w-right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    image = torch.from_numpy(image).float() / 127.5 - 1
    image = image.permute(2, 0, 1).unsqueeze(0)

    return image

def gray_code(n):
    """
    格雷码生成函数
    :param n: 格雷码位数
    :return: 格雷码列表
    """
    if n == 1:
        return ['0', '1']
    else:
        res = []
    old_gray_code = gray_code(n - 1)
    for i in range(len(old_gray_code)):
        res.append('0' + old_gray_code[i])
    for i in range(len(old_gray_code) - 1, -1, -1):
        res.append('1' + old_gray_code[i])
    return res


# 
# =============================================================
# === 
# ===          Hybrid ECC 輔助函數 (Hamming C(15, 11))
# ===          (附加於 2025/11/14)
# === 
# =============================================================

# --- Hamming C(15, 11) 參數 ---
# n=15 (總長度), k=11 (數據), r=4 (校驗)
# 可修正 1 個單比特錯誤

N_DATA = 11      # k
N_PARITY = 4     # r
N_CODEWORD = 15  # n

# 數據位和校驗位在 15 位碼字中的索引 (0-indexed)
# 校驗位 P 在 2^i 位置 (索引 0, 1, 3, 7)
PARITY_INDICES = [0, 1, 3, 7]
DATA_INDICES = [2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14]

# 校驗位 (P) 所需檢查的數據位 (D) 索引 (在 15 位碼字中的絕對索引)
# 這些是根據標準 Hamming(15,11) 的 H 矩陣計算得出的
P1_CHECKS = [2, 4, 6, 8, 10, 12, 14]
P2_CHECKS = [2, 5, 6, 9, 10, 13, 14]
P4_CHECKS = [4, 5, 6, 11, 12, 13, 14]
P8_CHECKS = [8, 9, 10, 11, 12, 13, 14]

CHECK_INDICES = [P1_CHECKS, P2_CHECKS, P4_CHECKS, P8_CHECKS]

def encode_c15_11(data_bits_11):
    """
    Hamming C(15, 11) Encoding: 11-bit data -> 15-bit codeword
    """
    if not isinstance(data_bits_11, np.ndarray):
        data_bits_11 = np.array(data_bits_11)
        
    assert len(data_bits_11) == N_DATA, f"Input must be {N_DATA} bits"
    codeword = np.zeros(N_CODEWORD, dtype=np.uint8)
    
    # 1. 將 11 個數據位放入碼字
    codeword[DATA_INDICES] = data_bits_11
    
    # 2. 計算 4 個校驗位 (Parity bits)
    for i, p_idx in enumerate(PARITY_INDICES):
        # 進行 XOR 運算 (sum mod 2)
        xor_sum = np.sum(codeword[CHECK_INDICES[i]]) % 2
        codeword[p_idx] = xor_sum
    
    return codeword


def decode_c15_11(received_codeword):
    """
    Hamming C(15, 11) Decoding: 修正 1 個錯誤
    """
    if not isinstance(received_codeword, np.ndarray):
        received_codeword = np.array(received_codeword)

    assert len(received_codeword) == N_CODEWORD, f"Input must be {N_CODEWORD} bits"
    
    # 複製一份，避免修改到原始數據
    codeword_copy = received_codeword.copy()
    
    # 1. 計算 Syndrome (症狀)
    syndrome_bits = []
    for i, p_idx in enumerate(PARITY_INDICES):
        # 檢查 P 位 和 P 應該檢查的 D 位
        indices_to_check = [p_idx] + CHECK_INDICES[i]
        xor_sum = np.sum(codeword_copy[indices_to_check]) % 2
        syndrome_bits.append(xor_sum)
        
    # 2. 查找錯誤位置
    # 將 syndrome [s8, s4, s2, s1] (H 矩陣順序) 轉換為十進制位置
    error_pos_int = (syndrome_bits[3] * 8 + 
                     syndrome_bits[2] * 4 + 
                     syndrome_bits[1] * 2 + 
                     syndrome_bits[0] * 1)
    
    errors_fixed = 0
    if error_pos_int > 0:
        # 索引是 0-indexed，所以位置 (1-15) 需減 1
        error_idx = error_pos_int - 1
        
        # 3. 修正錯誤：翻轉錯誤位
        if error_idx < N_CODEWORD:
            codeword_copy[error_idx] = 1 - codeword_copy[error_idx]
            errors_fixed = 1
    
    # 4. 提取 11 個數據位
    data_bits_11 = codeword_copy[DATA_INDICES]
    
    return data_bits_11, errors_fixed


def hamming_encode_stream(data_bits):
    """對整個比特流進行 C(15, 11) 編碼"""
    # 補零對齊 11 比特
    pad_len = (N_DATA - (len(data_bits) % N_DATA)) % N_DATA
    if pad_len == N_DATA: pad_len = 0 # 剛好整除
        
    data_bits_padded = np.pad(data_bits, (0, pad_len), 'constant', constant_values=0)
    
    encoded_stream = []
    for i in range(0, len(data_bits_padded), N_DATA):
        chunk = data_bits_padded[i:i+N_DATA]
        encoded_stream.append(encode_c15_11(chunk))
        
    return np.concatenate(encoded_stream), pad_len


def hamming_decode_stream(received_bits, original_pad_len):
    """對整個比特流進行 C(15, 11) 解碼"""
    decoded_stream = []
    total_fixed = 0
    
    # 確保長度是 15 的倍數，處理通道截斷問題
    if len(received_bits) % N_CODEWORD != 0:
        pad_len_stream = (N_CODEWORD - (len(received_bits) % N_CODEWORD)) % N_CODEWORD
        received_bits = np.pad(received_bits, (0, pad_len_stream), 'constant', constant_values=0)
        
    for i in range(0, len(received_bits), N_CODEWORD):
        codeword = received_bits[i:i+N_CODEWORD]
        data_bits, fixed = decode_c15_11(codeword)
        decoded_stream.append(data_bits)
        total_fixed += fixed
        
    data_bits_padded = np.concatenate(decoded_stream)
    
    # 移除 Alice 端添加的補零
    if original_pad_len > 0:
        data_bits = data_bits_padded[:-original_pad_len]
    else:
        data_bits = data_bits_padded
        
    return data_bits, total_fixed

# --- Hybrid ECC 輔助函數結束 ---