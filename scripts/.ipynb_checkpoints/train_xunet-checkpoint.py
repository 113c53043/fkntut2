# 檔案位置: scripts/train_xunet.py
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random

# 引入剛剛建立的模型定義
from xunet_model import XuNet

# 設定路徑
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(CURRENT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "training_data")
WEIGHTS_DIR = os.path.join(ROOT_DIR, "weights")
SAVE_PATH = os.path.join(WEIGHTS_DIR, "xunet_best.pth")

# === 數據集類別 ===
class StegoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.cover_dir = os.path.join(root_dir, "cover")
        self.stego_dir = os.path.join(root_dir, "stego")
        self.transform = transform
        
        self.covers = sorted(os.listdir(self.cover_dir))
        self.stegos = sorted(os.listdir(self.stego_dir))
        
        # 確保配對
        min_len = min(len(self.covers), len(self.stegos))
        self.covers = self.covers[:min_len]
        self.stegos = self.stegos[:min_len]
        self.data = []
        
        # 0: Cover, 1: Stego
        for f in self.covers:
            self.data.append((os.path.join(self.cover_dir, f), 0))
        for f in self.stegos:
            self.data.append((os.path.join(self.stego_dir, f), 1))
            
        random.shuffle(self.data) # 打亂順序

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except:
            # 錯誤處理：回傳隨機數據防止崩潰
            return torch.randn(1, 256, 256), label

# === 訓練流程 ===
def train():
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 預處理：Xu-Net 需要灰階圖 + 256x256
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    # 載入數據
    print("⏳ 正在加載數據...")
    if not os.path.exists(DATA_DIR):
        print(f"❌ 找不到數據資料夾 {DATA_DIR}，請先執行 gen_training_data.py")
        return

    full_dataset = StegoDataset(DATA_DIR, transform=transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=4)
    
    print(f"✅ 數據加載完成: 訓練集 {len(train_ds)} 張, 驗證集 {len(val_ds)} 張")

    # 初始化模型
    model = XuNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adamax(model.parameters(), lr=0.001) # 論文推薦 Adamax

    epochs = 15
    best_acc = 0.0

    print(f"🚀 開始訓練 (Epochs: {epochs})...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_acc = 100 * correct / total
        
        # 驗證
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  💾 保存最佳模型 -> {SAVE_PATH}")

    print("✅ 訓練結束！")

if __name__ == "__main__":
    train()