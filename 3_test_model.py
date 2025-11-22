import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np
import os
# --- 修正: 加入 PIL 庫的匯入 ---
from PIL import Image 
# ------------------------------------

# --- 1. 配置與參數設定 (需要與訓練時保持一致) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "trained_model"
CHECKPOINT_FILE = "latest_checkpoint.pth"
CHECKPOINT_PATH = os.path.join(MODEL_SAVE_PATH, CHECKPOINT_FILE)
NUM_CLASSES = 3  # nothing, hand, cup
CLASS_NAMES = ["nothing", "hand", "cup"] # 必須與模型訓練時的索引順序一致

# --- 2. 模型定義：CleanCNN (必須複製訓練腳本中的定義) ---
class CleanCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(CleanCNN, self).__init__()
        FINAL_CHANNELS = 64
        self.model = nn.Sequential(
            # 第一組 Conv + BN + ReLU (已修正，用於載入正確的權重結構)
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16), 
            nn.ReLU(), 
            nn.MaxPool2d(2),
            
            # 第二組 Conv + BN + ReLU
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), 
            nn.ReLU(), 
            nn.MaxPool2d(2),
            
            # 第三組 Conv + BN + ReLU (已移除冗餘 MaxPool)
            nn.Conv2d(32, FINAL_CHANNELS, kernel_size=3, padding=1),
            nn.BatchNorm2d(FINAL_CHANNELS), 
            nn.ReLU(), 
            nn.AdaptiveAvgPool2d(1), 
            
            nn.Flatten(),
            nn.Linear(FINAL_CHANNELS, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# --- 3. 影像預處理函式 ---
def get_transform():
    # 必須使用訓練時相同的 Normalization 參數
    return transforms.Compose([
        transforms.ToTensor(), # 將 (H, W, C) numpy array 轉換為 (C, H, W) Tensor，並將數值縮放到 [0, 1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

# --- 4. 載入模型函式 ---
def load_trained_model(path, num_classes):
    # 實例化模型
    model = CleanCNN(num_classes=num_classes).to(device)
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到檢查點檔案: {path}")

    # 載入整個檢查點字典
    checkpoint = torch.load(path, map_location=device)
    
    # 從字典中取出 model_state_dict 並載入
    state_dict = checkpoint.get('model_state_dict')
    if state_dict is None:
        raise KeyError("檢查點檔案中缺少 'model_state_dict' 鍵。")
        
    model.load_state_dict(state_dict)
    
    # 設置為推論模式 (關鍵！確保 BatchNorm 和 Dropout 表現正確)
    model.eval()
    print(f"成功載入模型權重: {path} (歷史最佳 Acc: {checkpoint.get('best_accuracy', 'N/A'):.2f}%)")
    return model

# --- 5. 主推論函式 ---
def main():
    try:
        # 載入模型
        model = load_trained_model(CHECKPOINT_PATH, NUM_CLASSES)
        data_transform = get_transform()

        # 啟動 WebCam (0 通常是預設相機)
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

            # 確保圖像顏色通道是 RGB (OpenCV 預設是 BGR)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # --- 影像預處理 ---
            # 1. 轉換為 PIL Image
            pil_image = Image.fromarray(rgb_frame) # <--- 使用匯入的 Image
            # 2. 轉換為 Tensor 並 Normalize
            input_tensor = data_transform(pil_image)
            # 3. 增加 Batch 維度 (C, H, W) -> (1, C, H, W)
            input_batch = input_tensor.unsqueeze(0).to(device) 

            # --- 推論 ---
            with torch.no_grad():
                output = model(input_batch)
            
            # --- 結果解碼 ---
            # 應用 Softmax 取得機率
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            
            # 找出最大機率的索引
            confidence, predicted_index = torch.max(probabilities, 0)
            
            predicted_class = CLASS_NAMES[predicted_index.item()]
            confidence_percent = confidence.item() * 100

            # --- 顯示結果 (使用 OpenCV) ---
            
            # 格式化輸出字串
            text = f"Class: {predicted_class} | Conf: {confidence_percent:.2f}%"
            
            # 將結果繪製到原始畫面 (BGR 格式)
            # 顏色使用綠色 (0, 255, 0)
            cv2.putText(frame, text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            
            # 顯示畫面
            cv2.imshow('Real-time Inference', frame)

            # 按 'q' 退出迴圈
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except FileNotFoundError as e:
        print(f"\n[錯誤] {e}")
        print("請確認您已執行訓練腳本並成功儲存了檢查點檔案。")
    except Exception as e:
        print(f"\n[致命錯誤] {e}")
    
    finally:
        # 釋放資源
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
