import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np
import os
import sys
from PIL import Image, ImageGrab # ImageGrab 用於剪貼簿
from typing import List, Optional, Any

# --- 導入模型結構 (假設 MobileNetTransfer 已在其中) ---
from model_defs import MobileNetTransfer 

# --- 1. 配置與參數設定 (必須與訓練時保持一致) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "trained_model"

# 使用 CIFAR-10 遷移學習的檢查點
CHECKPOINT_FILE = "latest_checkpoint_cifar10_mobilenet.pth"
CHECKPOINT_PATH = os.path.join(MODEL_SAVE_PATH, CHECKPOINT_FILE)

NUM_CLASSES = 10 
# CIFAR-10 類別名稱
CLASS_NAMES = [
    "plane", "car", "bird", "cat", "deer", 
    "dog", "frog", "horse", "ship", "truck"
]

# --- 輔助函式：生成提示畫面 ---
def create_info_image(text: str, size: tuple = (400, 600)) -> np.ndarray:
    """ 創建一個黑色背景，帶有指定文字的 OpenCV 圖像。 """
    height, width = size
    # 創建黑色圖像
    img = np.zeros((height, width, 3), dtype=np.uint8) 
    
    # 計算文字位置
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
    
    # 居中放置
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2 
    
    cv2.putText(img, text, (text_x, text_y), 
                font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
    
    return img

# --- 2. 影像預處理函式 ---
def get_transform():
    """獲取 MobileNetV2 推論標準化的前處理組合。"""
    return transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

# --- 3. 模型載入函式 (保持不變) ---
def load_model(num_classes: int):
    """載入 MobileNetTransfer 模型結構並載入權重。"""
    
    model = MobileNetTransfer(num_classes=num_classes, use_pretrained=False) 
    model.to(device)

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"錯誤: 找不到檢查點檔案 {CHECKPOINT_PATH}。請先運行訓練腳本。")
        
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ 成功載入模型權重 (Epoch: {checkpoint['epoch']}, Acc: {checkpoint['best_accuracy']:.2f}%)")
    except Exception as e:
        raise ValueError(f"錯誤: 載入模型權重失敗。請確認模型結構與檢查點是否匹配。\n錯誤訊息: {e}")

    model.eval() 
    return model

# --- 4. 圖片處理與推論函式 (修正 RGBA 轉換與字體縮放) ---
def inference_on_image(model: nn.Module, pil_image: Image.Image, transform: transforms.Compose, class_names: List[str]):
    """對 PIL Image 進行推論，並將結果繪製到圖片上。"""

    # 此處要注意clipboard裡取到的影像格式, 有些是RGBA, 有些是RGB.
    # 如果是RGBA, 會造成 RuntimeError: The size of tensor a (4) must match the size of tensor b (3)... 
    # 這個錯誤發生在 torchvision.transforms.functional.py 的 normalize 函式中，
    # 當它嘗試執行 tensor.sub_(mean).div_(std) (張量減去均值再除以標準差) 時。
    #   Tensor A (4): 指的是輸入張量的第一個維度 (通道數)。
    #                 當您從剪貼簿或某些 PNG 檔案中讀取圖片時，它們可能包含 4 個通道：R (紅)、G (綠)、B (藍) 和 A (Alpha，透明度)。
    #   Tensor B (3): 指的是您在 transforms.Normalize 中定義的 mean 和 std 列表的長度：mean=[0.485, 0.456, 0.406] (3個值)。
    # MobileNetV2 模型是針對 ImageNet 訓練的，ImageNet 圖片都是標準的 RGB 三通道圖像。
    # 當程式嘗試用 3 個值的 mean 去減去 4 個通道的輸入張量時，就會產生這個錯誤。
    # 解決方案：在將 PIL 圖像轉換為 NumPy 陣列之前，我們需要強制 PIL 圖像的格式為 RGB 三通道，即使它原本是 RGBA 四通道。

    # 強制將輸入圖片轉換為 RGB 三通道
    pil_image = pil_image.convert('RGB')
    
    # 1. 影像預處理
    input_tensor = transform(pil_image)
    input_batch = input_tensor.unsqueeze(0).to(device) 

    # 2. 推論
    with torch.no_grad():
        output = model(input_batch)
    
    # 3. 結果解碼
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    confidence, predicted_index = torch.max(probabilities, 0)
    
    predicted_class = class_names[predicted_index.item()]
    confidence_percent = confidence.item() * 100

    # 4. 轉換回 OpenCV 格式 (用於顯示)
    cv_image = np.array(pil_image) 
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

    # 5. 繪製結果
    text = f"Class: {predicted_class} | Conf: {confidence_percent:.2f}%"
    
    # 使用 cv_image 的形狀來計算 font_scale
    font_scale = max(0.6, cv_image.shape[0] / 500)
    
    cv2.putText(cv_image, text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2, cv2.LINE_AA)
    
    return cv_image, predicted_class, confidence_percent

# --- 5. 主執行區塊 (持續迴圈，增強剪貼簿處理) ---
def main():
    
    # --- 使用者輸入配置 ---
    IMAGE_PATH = "" 
    USE_CLIPBOARD = True
    # -----------------------
    
    try:
        # 步驟 1: 載入模型和前處理器
        model = load_model(NUM_CLASSES)
        data_transform = get_transform()
        
        display_image = create_info_image("Waiting for Input...") # 初始提示畫面
        last_clipboard_image_repr: Any = None # 用於追蹤剪貼簿內容是否改變/被處理
        
        print("\n--- 持續圖片辨識已啟動 (按 'q' 退出) ---")

        # 步驟 2: 持續迴圈進行推論
        while True:
            pil_image = None
            should_infer = False
            
            # A. 優先檢查檔案路徑 (如果設定了，且檔案存在)
            if IMAGE_PATH and os.path.exists(IMAGE_PATH):
                try:
                    # 檔案模式：每次重新讀取檔案
                    pil_image = Image.open(IMAGE_PATH)
                    should_infer = True
                except Exception:
                    # 如果檔案讀取失敗，顯示錯誤提示
                    display_image = create_info_image(f"File Error: {IMAGE_PATH}")
                    
            # B. 其次檢查剪貼簿
            elif USE_CLIPBOARD:
                try:
                    current_clipboard_content = ImageGrab.grabclipboard() 
                    
                    # 只有當內容與上次不同時才處理
                    if current_clipboard_content != last_clipboard_image_repr:
                        
                        last_clipboard_image_repr = current_clipboard_content
                        
                        # Case 1: 剪貼簿內容是 PIL Image (直接的圖像數據，如截圖)
                        if isinstance(current_clipboard_content, Image.Image):
                            pil_image = current_clipboard_content
                            should_infer = True
                            
                        # Case 2: 剪貼簿內容是檔案路徑列表 (如複製了圖片檔案)
                        elif isinstance(current_clipboard_content, list) and len(current_clipboard_content) > 0:
                            first_item = current_clipboard_content[0]
                            if isinstance(first_item, str) and os.path.isfile(first_item) and first_item.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                                pil_image = Image.open(first_item)
                                should_infer = True
                                print(f"🖼️ 載入檔案路徑: {first_item}   ", end='\r')
                                
                            else:
                                # 列表但不是圖片檔案
                                display_image = create_info_image(f"Clipboard Type: list (unsupported content)")
                                print(f"📝 剪貼簿內容: list (unsupported)   ", end='\r')
                                
                        # Case 3: 剪貼簿是其他類型 (如純文本、None)
                        else:
                            content_type_str = type(current_clipboard_content).__name__
                            display_image = create_info_image(f"Clipboard Type: {content_type_str}")
                            print(f"📝 剪貼簿內容: {content_type_str}   ", end='\r')
                            
                except Exception as e:
                    # 剪貼簿讀取錯誤，可能是權限或格式問題
                    display_image = create_info_image(f"Clipboard Read Error: {e.__class__.__name__}")
                    print(f"❌ 剪貼簿讀取錯誤: {e.__class__.__name__}   ", end='\r')
            
            # C. 如果需要推論 (可能是新載入的檔案或新貼的圖片)
            if should_infer and pil_image is not None:
                try:
                    result_cv_image, predicted_class, confidence_percent = inference_on_image(
                        model, pil_image, data_transform, CLASS_NAMES
                    )
                    
                    # 更新顯示圖片
                    display_image = result_cv_image
                    
                    # 在命令列輸出結果 (使用 \r 來覆蓋前一行)
                    print(f"👁️ 預測: {predicted_class} | 置信度: {confidence_percent:.2f}%   ", end='\r')
                
                except Exception as e:
                    # 推論過程中的錯誤 (例如圖片損壞)
                    display_image = create_info_image(f"Inference Error: {e.__class__.__name__}")
                    print(f"🔥 推論錯誤: {e.__class__.__name__}   ", end='\r')

            # D. 顯示結果
            cv2.imshow('Inference Result (Press "q" to quit)', display_image)
            
            # E. 檢查按鍵 (使用 cv2.waitKey(10) 確保足夠高的幀率和按鍵響應)
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break

    except (FileNotFoundError, ValueError) as e:
        print(f"\n[致命錯誤] {e}") 
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n[一般錯誤] 發生錯誤: {e}")
    
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()