# 檔案位置: scripts/xunet_model.py
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

# === KV Filter 預處理層 (核心) ===
class ImageProcessing(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # 定義固定的 KV 濾波器，用於提取隱寫雜訊
        kv_filter = (
            torch.tensor(
                [
                    [-1.0, 2.0, -2.0, 2.0, -1.0],
                    [2.0, -6.0, 8.0, -6.0, 2.0],
                    [-2.0, 8.0, -12.0, 8.0, -2.0],
                    [2.0, -6.0, 8.0, -6.0, 2.0],
                    [-1.0, 2.0, -2.0, 2.0, -1.0],
                ],
            ).view(1, 1, 5, 5)
            / 12.0
        )
        # 註冊為 buffer 以便隨模型移動到 GPU
        self.register_buffer('kv_filter', kv_filter)

    def forward(self, inp: Tensor) -> Tensor:
        return F.conv2d(inp, self.kv_filter, stride=1, padding=2)

# === 卷積區塊 ===
class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        activation: str = "relu",
        abs: bool = False,
    ) -> None:
        super().__init__()

        padding = 2 if kernel_size == 5 else 0
        self.activation = nn.Tanh() if activation == "tanh" else nn.ReLU()
        self.abs = abs
        
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.pool = nn.AvgPool2d(kernel_size=5, stride=2, padding=2)

    def forward(self, inp: Tensor) -> Tensor:
        if self.abs:
            return self.pool(self.activation(self.batch_norm(torch.abs(self.conv(inp)))))
        return self.pool(self.activation(self.batch_norm(self.conv(inp))))

# === Xu-Net 主模型 ===
class XuNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.preprocessing = ImageProcessing()
        self.layer1 = ConvBlock(1, 8, kernel_size=5, activation="tanh", abs=True)
        self.layer2 = ConvBlock(8, 16, kernel_size=5, activation="tanh")
        self.layer3 = ConvBlock(16, 32, kernel_size=1)
        self.layer4 = ConvBlock(32, 64, kernel_size=1)
        self.layer5 = ConvBlock(64, 128, kernel_size=1)
        
        # 使用 AdaptiveAvgPool2d 避免尺寸問題
        self.gap = nn.AdaptiveAvgPool2d(output_size=1)
        
        self.fully_connected = nn.Sequential(
            nn.Linear(in_features=128, out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=2),
            # 移除 LogSoftmax，直接輸出 Logits 供 CrossEntropyLoss 使用
        )

    def forward(self, image: Tensor) -> Tensor:
        out = self.preprocessing(image) # 使用 KV Filter
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.fully_connected(out)
        return out