import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
import cv2
import os
from PIL import Image
import urllib.request
from tqdm import tqdm
from typing import List

# --- 1. 配置與設定 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ImageNet 類別列表的 URL (這是 PyTorch 官方常用的類別列表)
IMAGENET_CLASSES_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
CLASSES_FILE_PATH = "imagenet_classes.txt"

# --- 2. 輔助函式：下載 ImageNet 類別列表 ---
def download_imagenet_classes(url: str, filepath: str) -> List[str]:
    """下載 ImageNet 類別名稱列表並載入。"""
    if not os.path.exists(filepath):
        print(f"找不到 {filepath}，正在從 {url} 下載...")
        try:
            with urllib.request.urlopen(url) as response, open(filepath, 'w') as out_file:
                # 簡單下載
                data = response.read().decode('utf-8')
                out_file.write(data)
            print("✅ 下載完成。")
        except Exception as e:
            print(f"❌ 下載失敗: {e}")
            raise
    
    # 載入類別名稱，每行一個
    with open(filepath, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    return classes

# --- 3. 模型載入函式 ---
def load_imagenet_mobilenet():
    """載入 MobileNetV2 預訓練模型 (1000 類別)。"""
    
    # 使用 DEFAULT 權重載入模型
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    
    # 將模型移至 GPU/CPU 並設置為推論模式
    model.to(device)
    model.eval()
    
    print("✅ MobileNetV2 預訓練模型載入完成 (1000 類別輸出)。")
    return model

# --- 4. 影像預處理函式 ---
def get_transform():
    """獲取 ImageNet 推論標準化的前處理組合。"""
    return transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(), 
        # ImageNet 標準化參數
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

# --- 5. 主推論函式 ---
def main():
    try:
        # 步驟 1: 下載並載入 ImageNet 類別名稱
        imagenet_classes = download_imagenet_classes(IMAGENET_CLASSES_URL, CLASSES_FILE_PATH)
        
        # 步驟 2: 載入模型和定義前處理
        model = load_imagenet_mobilenet()
        data_transform = get_transform()

        # 步驟 3: 啟動 WebCam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise IOError("無法打開 WebCam。請檢查相機連接或驅動程式。")

        print("\n--- 即時 ImageNet 原始類別推論已啟動 ---")
        print("按下 'q' 鍵退出。")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # --- 影像預處理 ---
            pil_image = Image.fromarray(rgb_frame)
            input_tensor = data_transform(pil_image)
            input_batch = input_tensor.unsqueeze(0).to(device) 

            # --- 推論 ---
            with torch.no_grad():
                output = model(input_batch)
            
            # --- 結果解碼 (1000 類別) ---
            # 應用 Softmax 取得機率
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            
            # 找出最大機率的索引和置信度
            confidence, predicted_index = torch.max(probabilities, 0)
            
            # 獲取 ImageNet 類別名稱
            predicted_class = imagenet_classes[predicted_index.item()]
            confidence_percent = confidence.item() * 100

            # --- 顯示結果 (使用 OpenCV) ---
            text = f"ImageNet Class: {predicted_class} | Conf: {confidence_percent:.2f}%"
            
            # 將結果繪製到原始畫面 (BGR 格式)
            cv2.putText(frame, text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            
            cv2.imshow('Real-time ImageNet Inference', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"\n[致命錯誤] {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 釋放資源
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()