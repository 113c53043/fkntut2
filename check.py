import os
from PIL import Image
from tqdm import tqdm
import time

# 設定路徑
BASE_DIR = "/nfs/Workspace/stt/mas_GRDH/outputs/paper_repro_results"
FOLDERS = ["cover_sd", "ours_pure", "ours_unc"]
TARGET_COUNT = 1000  # 目標保留數量

def cleanup_folder(folder_path):
    """保留最新的 TARGET_COUNT 張圖片，刪除舊的"""
    # 取得所有 png 檔案的完整路徑
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".png")]
    
    # 如果數量正常或偏少，就不動作
    if len(files) <= TARGET_COUNT:
        return

    print(f"🧹 Cleaning up {os.path.basename(folder_path)}: Found {len(files)} images, keeping newest {TARGET_COUNT}...")
    
    # 按修改時間排序 (最新的在前面，os.path.getmtime 數值越大越新)
    files.sort(key=os.path.getmtime, reverse=True)
    
    # 找出要刪除的檔案 (從第 1001 張開始以後的所有檔案)
    files_to_delete = files[TARGET_COUNT:]
    
    for f in files_to_delete:
        try:
            os.remove(f)
        except OSError as e:
            print(f"   ❌ Error deleting {f}: {e}")
            
    print(f"   ✅ Deleted {len(files_to_delete)} old images. Now holding {TARGET_COUNT} images.")

def check_folder(folder_name):
    path = os.path.join(BASE_DIR, folder_name)
    if not os.path.exists(path):
        print(f"❌ Missing folder: {path}")
        return

    # === 新增：先執行清理，刪除多餘舊圖 ===
    cleanup_folder(path)

    # 重新讀取清理後的檔案列表
    files = [f for f in os.listdir(path) if f.endswith(".png")]
    print(f"\n📂 Checking {folder_name} ({len(files)} images)...")
    
    corrupt_count = 0
    for f in tqdm(files):
        try:
            img_path = os.path.join(path, f)
            # 嘗試開啟並載入
            with Image.open(img_path) as img:
                img.verify() # 驗證結構
            
            # 再次開啟檢查是否全黑 (NaN Error 常見症狀)
            with Image.open(img_path) as img:
                extrema = img.convert("L").getextrema()
                if extrema == (0, 0): # 全黑
                    print(f"   ⚠️ Black Image detected: {f}")
                    corrupt_count += 1
                    
        except Exception as e:
            print(f"   ❌ Corrupt file: {f} ({e})")
            corrupt_count += 1
            
    if corrupt_count == 0:
        print(f"✅ {folder_name} is clean!")
    else:
        print(f"❌ {folder_name} has {corrupt_count} bad images.")

if __name__ == "__main__":
    for folder in FOLDERS:
        check_folder(folder)