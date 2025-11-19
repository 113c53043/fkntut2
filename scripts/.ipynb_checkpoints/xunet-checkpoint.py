import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import os

# === 1. 定義 Xu-Net 網絡架構 ===
class Abs(nn.Module):
    def __init__(self):
        super(Abs, self).__init__()
    def forward(self, x):
        return torch.abs(x)

class XuNet(nn.Module):
    def __init__(self):
        super(XuNet, self).__init__()
        
        # Group 1: 預處理與第一層卷積
        self.conv1 = nn.Conv2d(1, 8, kernel_size=5, padding=2, bias=False)
        self.abs = Abs()
        self.bn1 = nn.BatchNorm2d(8)
        self.tanh1 = nn.Tanh()
        self.pool1 = nn.AvgPool2d(kernel_size=5, stride=2, padding=2)
        
        # Group 2
        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.tanh2 = nn.Tanh()
        self.pool2 = nn.AvgPool2d(kernel_size=5, stride=2, padding=2)
        
        # Group 3
        self.conv3 = nn.Conv2d(16, 32, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.AvgPool2d(kernel_size=5, stride=2, padding=2)
        
        # Group 4
        self.conv4 = nn.Conv2d(32, 64, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.AvgPool2d(kernel_size=5, stride=2, padding=2)
        
        # Group 5
        self.conv5 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        self.bn5 = nn.BatchNorm2d(128)
        self.relu5 = nn.ReLU()
        # 【修正】使用 AdaptiveAvgPool2d((1, 1)) 來實現 Global Average Pooling
        self.pool5 = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully Connected Layer (分類器)
        self.fc = nn.Linear(128, 2) # 輸出 2 類: Cover(0) vs Stego(1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.abs(x)
        x = self.bn1(x)
        x = self.tanh1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.tanh2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.pool5(x)
        
        # 展平: (Batch, 128, 1, 1) -> (Batch, 128)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# === 2. 定義評估器工具類 ===
class XuNetEvaluator:
    def __init__(self, ckpt_path=None, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = XuNet().to(self.device)
        
        # 載入權重邏輯
        if ckpt_path and os.path.exists(ckpt_path):
            try:
                state_dict = torch.load(ckpt_path, map_location=self.device)
                # 處理不同儲存格式 (有些權重包在 'state_dict' key 裡)
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                
                self.model.load_state_dict(state_dict, strict=False)
                print(f"✅ [Xu-Net] 成功加載權重: {ckpt_path}")
            except Exception as e:
                print(f"⚠️ [Xu-Net] 加載權重失敗: {e} (將使用隨機權重)")
        else:
            print(f"⚠️ [Xu-Net] 未找到權重檔或路徑為空 ({ckpt_path})，使用隨機初始化模型 (僅供測試流程)。")
            
        self.model.eval()
        
        # 預處理: 轉灰階 -> Resize -> Tensor
        self.preprocess = transforms.Compose([
            transforms.Grayscale(num_output_channels=1), 
            transforms.Resize((256, 256)), 
            transforms.ToTensor(),
        ])

    def eval_image(self, img_path):
        """
        輸入圖片路徑，返回它是隱寫圖片(Stego)的機率 (0.0 ~ 1.0)
        """
        try:
            image = Image.open(img_path).convert('RGB')
            img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                logits = self.model(img_tensor)
                probs = F.softmax(logits, dim=1)
                # 假設 index 1 代表 Stego 類別
                stego_prob = probs[0, 1].item()
                
            return stego_prob
        except Exception as e:
            print(f"❌ [Xu-Net] 推理錯誤: {e}")
            return 0.5 # 失敗時回傳 0.5