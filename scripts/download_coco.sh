#!/bin/bash

# 設定目標檔案
ZIP_FILE="annotations_trainval2017.zip"
URL="http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
TARGET_DIR="coco_annotations"

if [ -f "$TARGET_DIR/annotations/captions_val2017.json" ]; then
    echo "✅ Captions 已經存在，跳過下載。"
    exit 0
fi

echo "🚀 下載 MS-COCO Annotations (包含 Prompts)..."
wget -c --no-check-certificate --show-progress -O "$ZIP_FILE" "$URL"

echo "📦 解壓縮..."
unzip -q "$ZIP_FILE"
mv annotations "$TARGET_DIR" # 將解壓出來的 annotations 資料夾移動並改名方便管理

echo "🧹 清理..."
rm "$ZIP_FILE"

echo "✅ 完成！Prompt 檔案位於: $TARGET_DIR/captions_val2017.json"