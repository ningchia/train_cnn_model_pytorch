import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import time
from typing import List, Tuple

# --- 導入 torchao 的量化 API ---
from torchao.quantization import quantize_ # 最上層的量化函式
from torchao.quantization import Int8DynamicActivationInt8WeightConfig # Int8 動態量化配置
# -----------------------------

# --- 從 model_defs 模組導入模型結構 ---
from model_defs import MobileNetTransfer 

# --- 1. 配置與參數設定 (與訓練/量化時保持一致) ---
# 測試通常在 CPU 上進行，以模擬部署環境
DEVICE = torch.device("cpu") 
DATA_DIR = "cifar10_data" 
MODEL_SAVE_PATH = "trained_model"
BATCH_SIZE = 64 # 測試時可以適當提高批次大小

# FP32 模型配置
FP32_CHECKPOINT_FILE = "latest_checkpoint_cifar10_mobilenet.pth"
FP32_CHECKPOINT_PATH = os.path.join(MODEL_SAVE_PATH, FP32_CHECKPOINT_FILE)

# INT8 模型配置
INT8_MODEL_FILE = "quantized_mobilenet_cifar10_int8.pth"
INT8_MODEL_PATH = os.path.join(MODEL_SAVE_PATH, INT8_MODEL_FILE)

NUM_CLASSES = 10 
# ImageNet 標準化參數 (MobileNetV2 標準輸入)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# --- 2. 數據加載 ---
def get_validation_loader(data_dir: str, batch_size: int) -> DataLoader:
    """載入 CIFAR-10 驗證集 DataLoader。"""
    
    # 驗證集專用轉換 (與訓練時驗證集的轉換必須一致)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD) 
    ])

    # 載入驗證集
    val_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=val_transform)

    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4 
    )
    
    return val_loader

# --- 3. 準確度計算函式 ---
def calculate_accuracy(loader: DataLoader, model: nn.Module) -> float:
    """ 計算模型在 DataLoader 上的準確度。 """
    
    # 確保模型在評估模式
    model.eval() 
    correct = 0
    total = 0
    
    # 禁用梯度計算
    with torch.no_grad():
        start_time = time.time()
        for images, labels in loader:
            # 推論時必須將資料移到模型所在的 DEVICE (這裡為 CPU)
            images, labels = images.to(DEVICE), labels.to(DEVICE) 
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        end_time = time.time()
        
    accuracy = 100 * correct / total
    inference_time = end_time - start_time
    
    return accuracy, inference_time


# --- 4. 模型載入函式：FP32 基準模型 ---
def load_fp32_model(num_classes: int) -> MobileNetTransfer:
    """載入 FP32 模型結構和權重。"""
    
    model = MobileNetTransfer(num_classes=num_classes, use_pretrained=False) 
    model.to(DEVICE)

    if not os.path.exists(FP32_CHECKPOINT_PATH):
        raise FileNotFoundError(f"錯誤: 找不到 FP32 檢查點檔案 {FP32_CHECKPOINT_PATH}。請先運行訓練腳本。")
        
    checkpoint = torch.load(FP32_CHECKPOINT_PATH, map_location=DEVICE)
    
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except Exception as e:
        raise ValueError(f"FP32 權重載入失敗: {e}")

    model.eval() 
    return model


# --- 5. 模型載入函式：INT8 量化模型 ---
def load_int8_model(num_classes: int) -> nn.Module:
    """載入 INT8 模型結構和權重。
    
    量化模型的載入流程是：
    1. 初始化原始 FP32 模型結構。
    2. 將該結構轉換為 INT8 量化結構 (使用 quantize_dynamic)。
    3. 載入保存的 INT8 權重 (state_dict) 到量化結構中。
    """
    
    if not os.path.exists(INT8_MODEL_PATH):
        raise FileNotFoundError(f"錯誤: 找不到 INT8 模型檔案 {INT8_MODEL_PATH}。請先運行量化腳本 12_quantize_model.py。")
        
    # 步驟 1: 初始化原始模型結構 (FP32)
    fp32_model = MobileNetTransfer(num_classes=num_classes, use_pretrained=False) 
    
    # 步驟 2: 定義量化配置 (與儲存時必須一致)
    quant_config = Int8DynamicActivationInt8WeightConfig()
    
    # 步驟 3: 將結構轉換為量化模型
    # 使用 quantize_ 進行 in-place 轉換
    quantize_(fp32_model, quant_config)
    quantized_model = fp32_model 
    
    # 步驟 4: 載入 INT8 權重
    int8_state_dict = torch.load(INT8_MODEL_PATH, map_location=DEVICE)

    try:
        quantized_model.load_state_dict(int8_state_dict)
    except Exception as e:
        raise ValueError(f"INT8 權重載入失敗。量化/載入結構可能不匹配: {e}")
        
    quantized_model.eval()
    return quantized_model


# --- 6. 主執行區塊 ---
def main():
    print("--- 啟動模型準確度與推論速度測試 (FP32 vs. INT8) ---")
    
    try:
        # 載入驗證集
        val_loader = get_validation_loader(DATA_DIR, BATCH_SIZE)
        print(f"✅ 載入 CIFAR-10 驗證集 (總樣本數: {len(val_loader.dataset)})")

        # 載入 FP32 模型
        print("\n⏳ 載入 FP32 基準模型...")
        fp32_model = load_fp32_model(NUM_CLASSES)
        
        # 載入 INT8 模型
        print("⏳ 載入 INT8 量化模型...")
        int8_model = load_int8_model(NUM_CLASSES)
        
        # --- 測試 FP32 模型 ---
        print("\n--- 測試 FP32 模型 ---")
        fp32_acc, fp32_time = calculate_accuracy(val_loader, fp32_model)
        
        # --- 測試 INT8 模型 ---
        print("--- 測試 INT8 模型 ---")
        int8_acc, int8_time = calculate_accuracy(val_loader, int8_model)
        
        # --- 輸出結果 ---
        print("\n" + "=" * 40)
        print("     🔥 模型量化效果分析 (CIFAR-10 驗證集) 🔥")
        print("=" * 40)
        
        # 準確度對比
        print(f"** 準確度 (Accuracy) **")
        print(f"FP32 模型準確度: {fp32_acc:.2f}%")
        print(f"INT8 模型準確度: {int8_acc:.2f}%")
        
        acc_drop = fp32_acc - int8_acc
        print(f"準確度損失 (Loss): {acc_drop:.2f}%")
        
        # 推論速度對比
        print(f"\n** 推論時間 (Inference Time) ** (總耗時)")
        print(f"FP32 模型推論總耗時: {fp32_time:.4f} 秒")
        print(f"INT8 模型推論總耗時: {int8_time:.4f} 秒")
        
        speed_up = fp32_time / int8_time if int8_time > 0 else float('inf')
        print(f"INT8 相較於 FP32 的加速比: {speed_up:.2f} 倍")
        print("=" * 40)
        
    except FileNotFoundError as e:
        print(f"\n[致命錯誤] {e}")
        print("請確保您已運行訓練腳本 (8_transfer_train_cifar10.py) 和量化腳本 (12_quantize_model.py)。")
    except ValueError as e:
        print(f"\n[致命錯誤] {e}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n[一般錯誤] 發生錯誤: {e}")

if __name__ == '__main__':
    main()