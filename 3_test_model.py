from model_defs import CleanCNN, MobileNetTransfer # 導入訓練/微調後的模型結構
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models # <--- 新增: 導入 torchvision models
import cv2
import numpy as np
import os
from typing import Literal

from PIL import Image 

# --- 1. 配置與參數設定 (需要與訓練時保持一致) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "trained_model"
NUM_CLASSES = 3  # nothing, hand, cup
CLASS_NAMES = ["nothing", "hand", "cup"] # 必須與模型訓練時的索引順序一致 (0, 1, 2)

MODEL_TO_TEST: Literal['clean_cnn', 'mobilenet_v2', 'mobilenet_v2_pretrained'] = 'mobilenet_v2_pretrained' # <--- 選擇要測試的模型

CHECKPOINT_FILE = "none"
if MODEL_TO_TEST == 'clean_cnn':
    CHECKPOINT_FILE = "latest_checkpoint.pth"
elif MODEL_TO_TEST in ['mobilenet_v2']:
    CHECKPOINT_FILE = "latest_checkpoint_mobilenet.pth"
# 預訓練模型不需要檢查點檔案

CHECKPOINT_PATH = os.path.join(MODEL_SAVE_PATH, CHECKPOINT_FILE)

# --- 2. ImageNet 類別映射定義 (用於 mobilenet_v2_pretrained 模式) ---
# 這是 ImageNet 1000 類別的預設索引到我們自訂 3 類別的映射。
# 注意：這些索引是常用的 ImageNet 索引範例，在實際應用中應查閱 MobileNetV2 權重
# 對應的 ImageNet 類別列表來確認準確的索引。
# 預設所有 ImageNet 類別都會被視為 'nothing' (索引 0)。
IMAGENET_TO_CUSTOM_MAPPING = {
    # ImageNet Index : 您的目標索引 (1: hand, 2: cup)
    425: 1,  # 假設 ImageNet index 425 是 'hand'
    504: 2,  # 假設 ImageNet index 504 是 'coffee mug'/'cup'
    # ... 您可以加入其他相關的 ImageNet 索引，例如不同的杯子或手勢。
}

# --- 3. 影像預處理函式 ---
def get_transform():
    # 統一將輸入影像縮放到 224x224 (MobileNetV2 標準輸入)，並使用 ImageNet 標準化參數
    return transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

# --- 4. 載入模型函式 ---
def load_trained_model(path, num_classes):
    
    # --- 情況 3: 載入預訓練 MobileNetV2 (1000類別) ---
    if MODEL_TO_TEST == 'mobilenet_v2_pretrained':
        # 載入 MobileNetV2 結構和 ImageNet 權重 (保留 1000 類別輸出)
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        model.to(device)
        model.eval()
        print("✅ 成功載入原始預訓練 MobileNetV2 模型 (1000 類別輸出)。")
        return model

    # --- 情況 1 & 2: 載入訓練過的模型 (3類別) ---
    if MODEL_TO_TEST == 'clean_cnn':
        model = CleanCNN(num_classes=num_classes).to(device)
    else:  # MODEL_TO_TEST == 'mobilenet_v2' (遷移學習後版本)
        # 這裡需要傳入 use_pretrained=False，因為我們會從檢查點載入訓練好的權重，
        # 避免再次下載和覆蓋 (雖然 load_state_dict 會覆蓋)
        model = MobileNetTransfer(num_classes=num_classes, use_pretrained=False).to(device)
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到檢查點檔案: {path}")

    # 載入檢查點
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get('model_state_dict')
    if state_dict is None:
        raise KeyError("檢查點檔案中缺少 'model_state_dict' 鍵。")
        
    model.load_state_dict(state_dict)
    
    # 設置為推論模式
    model.eval()
    print(f"✅ 成功載入訓練/微調後模型權重: {path} (歷史最佳 Acc: {checkpoint.get('best_accuracy', 'N/A'):.2f}%)")
    return model

# --- 5. 主推論函式 ---
def main():
    try:
        # 載入模型
        model = load_trained_model(CHECKPOINT_PATH, NUM_CLASSES)
        data_transform = get_transform()

        # 啟動 WebCam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise IOError("無法打開 WebCam。請檢查相機連接或驅動程式。")

        print("\n--- 即時推論已啟動 ---")
        print("按下 'q' 鍵退出。")

        while True:
            # 讀取一幀畫面
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
            
            # --- 結果解碼 ---
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            
            # 找出最大機率的索引
            confidence, predicted_index_raw = torch.max(probabilities, 0)
            
            # *** 針對預訓練模型的特殊處理：將 1000 個輸出映射到 3 個類別 ***
            if MODEL_TO_TEST == 'mobilenet_v2_pretrained':
                imagenet_index = predicted_index_raw.item()
                
                # 預設為 'nothing' (索引 0)
                predicted_index_custom = IMAGENET_TO_CUSTOM_MAPPING.get(imagenet_index, 0)
                predicted_index_final = predicted_index_custom
                
                # 為了避免誤判，如果最高機率的類別是我們關心的類別，我們使用它的機率。
                # 如果最高機率是其他類別 (被映射為 nothing)，我們仍然使用那個最高機率作為 confidence
                # 但類別名稱顯示 'nothing'。
            else:
                # CleanCNN 或訓練/微調後的 MobileNetV2 (3 類別輸出)
                predicted_index_final = predicted_index_raw.item()
                
            predicted_class = CLASS_NAMES[predicted_index_final]
            confidence_percent = confidence.item() * 100

            # --- 顯示結果 (使用 OpenCV) ---
            text = f"Class: {predicted_class} | Conf: {confidence_percent:.2f}%"
            
            cv2.putText(frame, text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            
            cv2.imshow('Real-time Inference', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except FileNotFoundError as e:
        print(f"\n[錯誤] {e}")
        print("請確認您已執行訓練腳本並成功儲存了檢查點檔案。")
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