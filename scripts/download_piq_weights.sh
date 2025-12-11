#!/bin/bash

# 1. 建立 PyTorch Hub 的快取目錄
CACHE_DIR="$HOME/.cache/torch/hub/checkpoints"
mkdir -p "$CACHE_DIR"

# 2. 定義權重檔網址 (piq 官方來源)
WEIGHTS_URL="https://github.com/photosynthesis-team/piq/releases/download/v0.4.0/brisque_svm_weights.pt"
TARGET_FILE="$CACHE_DIR/brisque_svm_weights.pt"

# 3. 下載 (略過 SSL 驗證)
echo "🚀 Downloading BRISQUE weights for piq..."
wget --no-check-certificate --show-progress -O "$TARGET_FILE" "$WEIGHTS_URL"

if [ -f "$TARGET_FILE" ]; then
    echo "✅ Success! Weights saved to: $TARGET_FILE"
    echo "   Now you can run the python script without download errors."
else
    echo "❌ Download failed."
fi