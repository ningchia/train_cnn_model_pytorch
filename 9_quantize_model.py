import torch
import torch.nn as nn
import os
import sys
import time
from typing import Literal

# --- 導入 torchao 的量化 API ---
from torchao.quantization import quantize_ # 最上層的量化函式
from torchao.quantization import Int8DynamicActivationInt8WeightConfig # Int8 動態量化配置
# -----------------------------

# 假設 MobileNetTransfer 結構定義在 cnn_models.py 或 model_defs.py 中
# 這裡我們需要導入它
# from model_defs import MobileNetTransfer 
# 為了讓腳本獨立運行，我們可以在此重新定義，但建議使用導入。
# 假設您已將 MobileNetTransfer 放在 cnn_models.py 或 model_defs.py 中
try:
    from model_defs import MobileNetTransfer 
except ImportError:
    # 如果導入失敗，這裡放入一個臨時的結構定義（需與訓練時保持一致）
    # 請確保您的 MobileNetTransfer 結構與訓練時一致
    print("WARNING: model_defs.MobileNetTransfer 導入失敗，請確保文件存在。")
    # 這裡省略了 MobileNetTransfer 的完整定義，假設它已被正確導入。
    pass


# --- 1. 配置與參數設定 (必須與訓練時保持一致) ---
MODEL_SAVE_PATH = "trained_model"
# 輸入的 FP32 檢查點
FP32_CHECKPOINT_FILE = "latest_checkpoint_cifar10_mobilenet.pth"
FP32_CHECKPOINT_PATH = os.path.join(MODEL_SAVE_PATH, FP32_CHECKPOINT_FILE)

# 輸出的 INT8 模型檔案名稱
INT8_MODEL_FILE = "quantized_mobilenet_cifar10_int8.pth"
INT8_MODEL_PATH = os.path.join(MODEL_SAVE_PATH, INT8_MODEL_FILE)

NUM_CLASSES = 10 
DEVICE = torch.device("cpu") # 量化通常在 CPU 上進行，且 INT8 模型主要用於 CPU 部署

# --- 2. 模型載入函式 ---
def load_fp32_model(num_classes: int) -> MobileNetTransfer:
    """載入 MobileNetTransfer 模型結構並載入 FP32 權重。"""
    
    # 必須使用 use_pretrained=False，因為我們要載入本地訓練好的權重
    model = MobileNetTransfer(num_classes=num_classes, use_pretrained=False) 
    model.to(DEVICE)

    if not os.path.exists(FP32_CHECKPOINT_PATH):
        raise FileNotFoundError(f"錯誤: 找不到 FP32 檢查點檔案 {FP32_CHECKPOINT_PATH}。請先運行訓練腳本。")
        
    checkpoint = torch.load(FP32_CHECKPOINT_PATH, map_location=DEVICE)
    
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ 成功載入 FP32 模型權重 (Epoch: {checkpoint['epoch']}, Acc: {checkpoint['best_accuracy']:.2f}%)")
    except Exception as e:
        raise ValueError(f"錯誤: 載入模型權重失敗。請確認模型結構與檢查點是否匹配。\n錯誤訊息: {e}")

    model.eval() # 設定為評估模式
    return model

# --- 3. 量化流程主函式 ---
def quantize_model():
    print("--- 啟動模型量化流程 ---")
    
    try:
        # 步驟 1: 載入 FP32 模型
        fp32_model = load_fp32_model(NUM_CLASSES)
        
        # 步驟 2: 定義量化配置 (Int8 Dynamic Quantization)
        # 此配置會將權重轉為 Int8，並在推論時動態量化激活值。
        quant_config = Int8DynamicActivationInt8WeightConfig()

        # 步驟 3: 執行後訓練動態量化 (Post-Training Dynamic Quantization)
        # 這種方法會將權重從 FP32 轉換為 INT8，並在推論時動態校準激活值。
        print("\n⏳ 正在執行後訓練動態量化 (FP32 -> INT8) 使用 quantize_ 函式...")
        start_time = time.time()
        
        # **主要修改點**: 使用 quantize_ 搭配配置
        # quantize_ 是 in-place 函式，會直接修改 fp32_model
        quantize_(fp32_model, quant_config)
        
        quantized_model = fp32_model 
        
        end_time = time.time()
        print(f"✅ 量化完成！耗時: {end_time - start_time:.2f} 秒")
        
        # 步驟 4: 儲存量化後的 INT8 模型
        torch.save(quantized_model.state_dict(), INT8_MODEL_PATH)

        # 步驟 5: 驗證檔案大小和準確度差異 (可選，但強烈推薦)
        fp32_size = os.path.getsize(FP32_CHECKPOINT_PATH)
        int8_size = os.path.getsize(INT8_MODEL_PATH)

        print("-" * 40)
        print(f"FP32 模型大小: {fp32_size / (1024**2):.2f} MB")
        print(f"INT8 模型大小: {int8_size / (1024**2):.2f} MB")
        print(f"檔案大小縮減比例: {fp32_size / int8_size:.2f} 倍")
        print(f"\n🎉 INT8 模型已成功儲存到: {INT8_MODEL_PATH}")
        print("-" * 40)
        
        # 注意：要測試 INT8 模型的實際推論準確度，需要使用專門的 INT8 模型載入和測試腳本。
        
    except FileNotFoundError as e:
        print(f"\n[錯誤] {e}")
        print("請確認 FP32 檢查點路徑是否正確。")
    except ValueError as e:
        print(f"\n[錯誤] {e}")
        print("請確認 MobileNetTransfer 類別的定義是否正確。")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n[一般錯誤] 量化過程中發生錯誤: {e}")

if __name__ == '__main__':
    quantize_model()