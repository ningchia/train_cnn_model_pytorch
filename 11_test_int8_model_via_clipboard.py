import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import numpy as np
import os
import sys
import time
from PIL import Image, ImageGrab 
from typing import List, Optional, Any, Tuple, Literal

# --- 導入模型結構 (假設 MobileNetTransfer 已在 model_defs 中) ---
# 確保 model_defs 檔案包含 MobileNetTransfer 類別
try:
    from model_defs import MobileNetTransfer 
except ImportError:
    print("錯誤: 找不到 model_defs 模組或 MobileNetTransfer 類別。請檢查檔案。")
    sys.exit(1)

# --- 導入 torchao 的量化 API ---
try:
    from torchao.quantization import quantize_ 
    from torchao.quantization import Int8DynamicActivationInt8WeightConfig 
except ImportError:
    quantize_ = None
    Int8DynamicActivationInt8WeightConfig = None
    
# --- 1. 配置與參數設定 ---
MODEL_SAVE_PATH = "trained_model"
INT8_MODEL_FILE = "quantized_mobilenet_cifar10_int8.pth"
INT8_MODEL_PATH = os.path.join(MODEL_SAVE_PATH, INT8_MODEL_FILE)

NUM_CLASSES = 10 
CLASS_NAMES = [
    "plane", "car", "bird", "cat", "deer", 
    "dog", "frog", "horse", "ship", "truck"
]

# 🌟 設置點 1: 選擇推論裝置 (可選 'cpu' 或 'cuda') 🌟
# 注意：若選擇 'cuda' 且安裝了 PyTorch 2.x+，模型會透過 torch.compile 進行優化。
# 否則，INT8 推論在 CUDA 上可能會失敗或效率極低。
INFERENCE_DEVICE: Literal['cpu', 'cuda'] = 'cpu' 
DEVICE = torch.device(INFERENCE_DEVICE)

# --- 輔助函式：生成提示畫面 (與原腳本相同) ---
def create_info_image(text: str, size: tuple = (400, 600)) -> np.ndarray:
    """ 創建一個黑色背景，帶有指定文字的 OpenCV 圖像。 """
    height, width = size
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # ... (其餘文字繪製邏輯略) ...
    cv2.putText(img, text, (10, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return img

# --- 輔助函式：PIL 轉 OpenCV (與原腳本相同) ---
def pil_to_cv2_with_text(pil_image: Image.Image, text: str) -> np.ndarray:
    # ... (程式碼略) ...
    cv_image = np.array(pil_image)
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
    cv2.putText(cv_image, text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    return cv_image

# --- 輔助函式：數據轉換 (與原腳本相同) ---
def create_data_transform() -> transforms.Compose:
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    return transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD) 
    ])

# --- 2. 核心：載入 INT8 模型函式 (支援 CPU/GPU 推論) ---
def load_int8_model(num_classes: int) -> nn.Module:
    """載入 INT8 模型並將其移動到指定的 DEVICE。"""
        
    if quantize_ is None:
         raise ImportError("錯誤: 未找到 torchao 函式庫。請運行 pip install torchao。")
    
    if not os.path.exists(INT8_MODEL_PATH):
        raise FileNotFoundError(f"錯誤: 找不到 INT8 模型檔案 {INT8_MODEL_PATH}。請先進行量化。")

    # 步驟 1: 初始化 FP32 模型結構 (作為量化的起點)
    model = MobileNetTransfer(num_classes=num_classes, use_pretrained=False).to(torch.device('cpu')) 
    
    # 步驟 2: 轉換為 INT8 量化結構
    quant_config = Int8DynamicActivationInt8WeightConfig()
    quantize_(model, quant_config) # In-place 轉換
    
    # 步驟 3: 載入 INT8 權重
    int8_state_dict = torch.load(INT8_MODEL_PATH, map_location=torch.device('cpu'))
    
    try:
        model.load_state_dict(int8_state_dict) 
    except Exception as e:
        raise ValueError(f"錯誤: 載入 INT8 權重失敗。結構可能不匹配。\n訊息: {e}")

    # 步驟 4: 移動模型到推論裝置並應用優化
    if INFERENCE_DEVICE == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError("錯誤: 已選擇 'cuda' 推論，但未偵測到 CUDA 裝置。")
            
        # ⚠️ 必須先將模型移動到 CUDA 裝置
        model.to(DEVICE)
        
        # ⚠️ 使用 torch.compile 進行 GPU 量化推論優化 (繞過 nn.quantized 模組限制)
        if torch.__version__ >= '2.0':
            print("⏳ 正在對 INT8 模型執行 torch.compile 優化 (初次運行較慢)...")
            model = torch.compile(model)
        else:
            print("警告: PyTorch 版本低於 2.0，無法使用 torch.compile 進行 GPU INT8 優化。推論可能效率低下。")

    elif INFERENCE_DEVICE == 'cpu':
        # 確保模型在 CPU 上 (它已經在 CPU 上了)
        pass 

    model.eval() # 設定為評估模式
    print(f"✅ INT8 模型已載入並部署於 {DEVICE}。")
    return model

# --- 3. 推論核心 ---
def inference_on_image(model: nn.Module, pil_image: Image.Image, data_transform: transforms.Compose, class_names: List[str], device: torch.device) -> Tuple[np.ndarray, str, float, float]:
    """ 執行推論，並返回結果、預測類別、置信度和推論時間。 """
    
    # 1. 執行轉換
    input_tensor = data_transform(pil_image)
    # 2. 增加 Batch 維度 (C, H, W) -> (1, C, H, W)，並移動到裝置
    input_batch = input_tensor.unsqueeze(0).to(device) 

    # --- 推論 ---
    with torch.no_grad():
        start_time = time.time()
        output = model(input_batch)
        end_time = time.time()
    
    inference_time = (end_time - start_time) * 1000 # 轉為毫秒
    
    # --- 結果解碼 ---
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    confidence, predicted_index = torch.max(probabilities, 0)
    
    predicted_class = class_names[predicted_index.item()]
    confidence_percent = confidence.item() * 100

    # 格式化輸出字串
    text = f"Class: {predicted_class} | Conf: {confidence_percent:.2f}% | Time: {inference_time:.2f}ms"
    
    # 將結果繪製到原始畫面 (BGR 格式)
    result_cv_image = pil_to_cv2_with_text(pil_image, text)
    
    return result_cv_image, predicted_class, confidence_percent, inference_time


# --- 4. 主執行流程 ---
def main():
    print(f"--- 啟動 INT8 模型實時推論 (裝置: {INFERENCE_DEVICE}) ---")
    
    try:
        # 載入 INT8 模型
        model = load_int8_model(NUM_CLASSES)
        data_transform = create_data_transform()

        # 變數用於追蹤剪貼簿狀態
        last_image_hash: Optional[int] = None
        display_text = f"Ready: INT8 on {INFERENCE_DEVICE}. Paste image to infer."
        display_image: np.ndarray = create_info_image(display_text)
        
        while True:
            should_infer = False
            pil_image: Optional[Image.Image] = None
            
            # A. 嘗試從剪貼簿讀取圖像
            try:
                current_pil_image = ImageGrab.grabclipboard()
                
                if current_pil_image is not None and isinstance(current_pil_image, Image.Image):
                    # 檢查圖像是否改變
                    current_image_hash = hash(current_pil_image.tobytes())
                    
                    if current_image_hash != last_image_hash:
                        pil_image = current_pil_image
                        last_image_hash = current_image_hash
                        should_infer = True
                        # print("\n✨ 偵測到剪貼簿圖像更新，正在推論...") # 避免頻繁輸出
                
            except Exception as e:
                # 處理剪貼簿讀取錯誤
                display_image = create_info_image("ERROR: Cannot read clipboard.")
                # print(f"🔥 剪貼簿讀取錯誤: {e.__class__.__name__}   ", end='\r')
            
            # B. 如果需要推論
            if should_infer and pil_image is not None:
                try:
                    result_cv_image, predicted_class, confidence_percent, time_ms = inference_on_image(
                        model, pil_image, data_transform, CLASS_NAMES, DEVICE
                    )
                    
                    # 更新顯示圖片
                    display_image = result_cv_image
                    
                    # 在命令列輸出結果
                    print(f"👁️ 預測: {predicted_class} | 置信度: {confidence_percent:.2f}% | 耗時: {time_ms:.2f}ms   ", end='\r')
                
                except Exception as e:
                    # 推論過程中的錯誤
                    display_image = create_info_image(f"Inference Error: {e.__class__.__name__}")
                    print(f"🔥 推論錯誤: {e.__class__.__name__}   ", end='\r')

            # C. 顯示結果
            cv2.imshow(f'INT8 Inference ({INFERENCE_DEVICE.upper()} Mode - Press "q" to quit)', display_image)
            
            # D. 檢查按鍵
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
        
    except (FileNotFoundError, ValueError, ImportError, RuntimeError) as e:
        print(f"\n\n[致命錯誤] {e}")
        cv2.destroyAllWindows()
        sys.exit(1)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n\n[一般錯誤] 發生錯誤: {e.__class__.__name__}")
        cv2.destroyAllWindows()
        sys.exit(1)
        
    finally:
        cv2.destroyAllWindows()
        print("\n\n👋 程式結束。")

if __name__ == '__main__':
    main()