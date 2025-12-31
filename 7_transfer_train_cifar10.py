import torch
import torch.nn as nn
# --- 導入模型結構: 使用 MobileNetTransfer ---
from model_defs import MobileNetTransfer 

import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
import time
from PIL import Image
import warnings

import numpy as np
import random
from typing import List, Tuple

# 忽略 PIL/Image 庫可能發出的警告
warnings.filterwarnings("ignore", category=UserWarning)

# --- 1. 配置與參數設定 (已更新) ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "cifar10_data" # 將數據下載到此資料夾
MODEL_SAVE_PATH = "trained_model"
CHECKPOINT_FILE = "latest_checkpoint_cifar10_mobilenet.pth" # 更改檢查點檔案名稱
NUM_EPOCHS = 50 # <-- 僅訓練 50 個 Epochs
BATCH_SIZE = 32
# 遷移學習時，只訓練分類頭部，使用較高的學習率
TRANSFER_LEARNING_LR = 0.001 
# FINE_TUNE_LR = 0.00001 # 這裡僅做分類頭訓練，暫不使用微調 LR

WANT_REPRODUCEBILITY = False    # 是否要強化訓練結果的可重現性 (Reproducibility)
SEED = 42
USE_PRETRAINED = True           # 必須使用預訓練權重

# ImageNet 標準化參數 (MobileNetV2 標準輸入)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# CIFAR-10 類別名稱 (10 個類別)
CIFAR10_CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
NUM_CLASSES = len(CIFAR10_CLASSES)


def set_seed(seed_value=42):
    """ 設定所有隨機性的種子，確保結果可重現。 """
    print("設定所有隨機性的種子，確保結果可重現。")
    random.seed(seed_value)         
    np.random.seed(seed_value)      
    torch.manual_seed(seed_value)   
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)      
        torch.backends.cudnn.deterministic = True   
        torch.backends.cudnn.benchmark = False      
    
    def seed_worker(worker_id):
        worker_seed = seed_value + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        
    g = torch.Generator()
    g.manual_seed(seed_value)
    
    return seed_worker, g # 返回 worker_init_fn 和 generator 給 DataLoader 使用


# --- 2. 輔助函式：凍結/解凍模型基礎層 ---
def freeze_base_layers(model: MobileNetTransfer, freeze: bool):
    """ 凍結或解凍 MobileNetV2 的基礎特徵提取層。 """
    # base_model 是 MobileNetV2 的特徵提取部分
    for param in model.base_model.features.parameters():
        param.requires_grad = freeze
    
    # 分類器頭部 (classifier) 保持可訓練
    for param in model.base_model.classifier.parameters():
        param.requires_grad = True

    if freeze:
        print("💡 模型基礎特徵提取層已凍結 (只訓練分類器頭部)。")
    else:
        print("💡 模型基礎特徵提取層已解凍 (準備進行微調/Fine-tuning)。")


# --- 3. 數據加載：使用內建 CIFAR-10 數據集 ---

def get_loaders(data_dir: str, batch_size: int, want_reproducibility: bool, seed: int) -> Tuple[DataLoader, DataLoader, int, List[str]]:
    """載入 CIFAR-10 數據集並回傳 DataLoader 和類別資訊。"""
    
    # 訓練集專用轉換 (Resize 到 224x224, 包含數據擴增)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD) 
    ])

    # 驗證集專用轉換 
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD) 
    ])

    worker_init_fn = None
    generator = None
    if want_reproducibility:
        worker_init_fn, generator = set_seed(seed)
    
    # 在 PyTorch 中，所有的 torchvision.datasets 物件在實例化後，都會內建兩個非常重要的屬性：.classes (清單) 和 .class_to_idx (字典)。
    # 若要知道 CIFAR-10 的類別名稱和索引對應關係，不需要去讀源碼，只需要在 Python 交互式環境（如 Jupyter 或 Python REPL）跑下面程式即可：
    #   from torchvision import datasets
    #   train_data = datasets.CIFAR10(root="data", train=True, download=True)
    #   print(train_data.classes)       # 印出標籤字串清單
    #   print(train_data.class_to_idx)  # 印出 {標籤: 索引} 的對照表
    #
    # 另外無論是torch, tensorflow/keras, 都可以使用python內建的 dir()函式來查看 dataset 物件的所有屬性和方法：
    #   print(dir(train_data))

    # 載入訓練集
    # 可以用下面方式看看一個sample有哪些欄位. 
    #   img, label = dataset[0]
    #   print(f"Image shape: {getattr(img, 'size', 'N/A')}, Label: {label}")    # getattr(object, name[, default])
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    # 載入驗證集
    val_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=val_transform)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4, 
        worker_init_fn=worker_init_fn if want_reproducibility else None, 
        generator=generator if want_reproducibility else None
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        generator=generator if want_reproducibility else None
    )

    return train_loader, val_loader, NUM_CLASSES, list(CIFAR10_CLASSES)


def calculate_accuracy(loader, model):
    # 保持不變
    model.eval() 
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


def save_checkpoint(epoch, model, optimizer, best_acc, path):
    # 保持不變
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_accuracy': best_acc,
        'timestamp': time.strftime("%Y%m%d-%H%M%S")
    }
    torch.save(checkpoint, path)
    print(f"\n[CHECKPOINT] 狀態已儲存到 {path} (Epoch: {epoch}, Acc: {best_acc:.2f}%)")

def load_checkpoint(path, model, optimizer):
    # 保持不變
    if not os.path.exists(path):
        return 0, 0.0, False 

    checkpoint = torch.load(path, map_location=device)
    
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_accuracy = checkpoint.get('best_accuracy', 0.0)
        
        print(f"\n[CHECKPOINT] 已載入檢查點，從 Epoch {start_epoch} 恢復訓練 (歷史最佳 Acc: {best_accuracy:.2f}%)")
        return start_epoch, best_accuracy, True
        
    except Exception as e:
        print(f"[警告] 檢查點載入失敗: {e}。將從頭開始訓練。")
        return 0, 0.0, False

# --- 4. 訓練流程主函式 (已修正優化器和 LR 邏輯) ---
def train_model(train_loader, val_loader, model, total_epochs, start_epoch, initial_best_acc):
    
    # 由於基礎層凍結，優化器只優化 requires_grad=True (即分類器頭部) 的參數
    # 使用 TRANSFER_LEARNING_LR 進行訓練
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=TRANSFER_LEARNING_LR
    )
    
    criterion = nn.CrossEntropyLoss()
    
    best_accuracy = initial_best_acc 
    visual_best_acc = best_accuracy
    
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    checkpoint_path = os.path.join(MODEL_SAVE_PATH, CHECKPOINT_FILE)
    
    # --- 儲存邏輯的起始 Epoch ---
    # 由於只有 50 個 Epoch，在最後 10 個 Epoch 開始儲存
    saving_start_epoch = total_epochs - 10 

    # 如果是從檢查點恢復訓練，需要重新設定學習率
    if start_epoch > 0:
        for param_group in optimizer.param_groups:
            # 確保使用正確的遷移學習 LR
            param_group['lr'] = TRANSFER_LEARNING_LR
        print(f"[續訓] 設置當前學習率為 {optimizer.param_groups[0]['lr']}")
    
    print(f"\n--- 開始遷移學習 (總目標 Epoch: {total_epochs}, 從 Epoch {start_epoch + 1} 開始) ---")
    print(f"訓練模式: 基礎層已凍結，只訓練分類器頭部 (LR={TRANSFER_LEARNING_LR})。")
    print(f"注意: 模型儲存功能將在第 {saving_start_epoch + 1} 個 Epoch 啟動。")
    
    try:
        for epoch in range(start_epoch, total_epochs):
            current_epoch_num = epoch + 1
            
            # 確保在訓練開始時模型是凍結狀態 (只訓練新加的分類頭部)
            model.train() 
            
            # --- 訓練階段 ---
            running_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {current_epoch_num}/{total_epochs}", leave=False)
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)
                pbar.set_postfix({'loss': loss.item()})

            train_loss = running_loss / len(train_loader.dataset)
            
            # --- 驗證階段 ---
            model.eval() 
            val_loss = 0.0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * images.size(0)
            
            val_loss /= len(val_loader.dataset)
            val_accuracy = calculate_accuracy(val_loader, model)
            
            # --- 儲存檢查點邏輯 ---
            print_message = f"Epoch {current_epoch_num}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%"
            
            if val_accuracy > visual_best_acc:
                visual_best_acc = val_accuracy
                print_message += f" (新歷史最高: {visual_best_acc:.2f}%)"
            
            is_saving_epoch = current_epoch_num >= saving_start_epoch
            
            if is_saving_epoch:
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    save_checkpoint(epoch, model, optimizer, best_accuracy, checkpoint_path)
                    print_message += f" -> **模型狀態已更新儲存** (目前最佳 Acc: {best_accuracy:.2f}%)"
                else:
                    print_message += f" (儲存區間內，目前最佳: {best_accuracy:.2f}%)"
            
            else:
                print_message += f" (儲存功能關閉，剩餘 {saving_start_epoch - current_epoch_num} 個 Epoch 啟動)"

            print(print_message)
            
    except KeyboardInterrupt:
        print("\n\n*** [使用者中斷] 偵測到 Ctrl+C，提前結束訓練。 ***")
        try:
             save_checkpoint(epoch, model, optimizer, best_accuracy, checkpoint_path)
        except NameError:
             print("-> 無法儲存，因為異常發生在第一個 Epoch 開始之前。")
        except Exception as save_err:
             print(f"-> 儲存檢查點時發生錯誤: {save_err}")

    except RuntimeError as e:
        if "DataLoader worker" in str(e):
             print("\n\n*** [DataLoader中斷] 偵測到 DataLoader worker 異常退出 (可能由 Ctrl+C 引起)。 ***")
             try:
                 save_checkpoint(epoch, model, optimizer, best_accuracy, checkpoint_path)
                 print(f"-> 成功儲存檢查點，以防數據丟失。")
             except NameError:
                 print("-> 無法儲存，因為異常發生在第一個 Epoch 開始之前。")
             except Exception as save_err:
                 print(f"-> 儲存檢查點時發生錯誤: {save_err}")
        else:
            raise e
            
    finally:
        print("-" * 50)
        print(f"訓練流程結束。")
        print(f"整體訓練過程中的最高準確度: {visual_best_acc:.2f}%\n")
        if best_accuracy > 0.0:
            print(f"最終儲存的最佳準確度: {best_accuracy:.2f}%")
            
# --- 5. 執行區塊 ---
if __name__ == '__main__':
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    checkpoint_path = os.path.join(MODEL_SAVE_PATH, CHECKPOINT_FILE)

    try:
        train_loader, val_loader, num_classes_detected, class_names_detected = get_loaders(
            DATA_DIR, BATCH_SIZE, WANT_REPRODUCEBILITY, SEED
        )
        
        print(f"總訓練樣本數: {len(train_loader.dataset)}")
        print(f"總驗證樣本數: {len(val_loader.dataset)}")
        print(f"偵測到類別數量: {num_classes_detected}")
        print(f"類別名稱: {class_names_detected}")
        
        # 步驟 1: 初始化模型 (使用預訓練 MobileNetV2)
        model = MobileNetTransfer(num_classes=num_classes_detected, use_pretrained=USE_PRETRAINED).to(device)
        # 初始化優化器（用於 load_checkpoint 載入狀態，使用遷移學習 LR）
        initial_optimizer = optim.Adam(model.parameters(), lr=TRANSFER_LEARNING_LR) 

        # 步驟 2: 載入檢查點
        start_epoch, best_accuracy, is_resumed = load_checkpoint(checkpoint_path, model, initial_optimizer)
        
        # 步驟 3: 確保模型凍結狀態正確 (遷移學習的關鍵步驟)
        # 在只訓練分類頭部的階段，確保基礎層是凍結的
        freeze_base_layers(model, freeze=True)

        # 步驟 4: 開始訓練 
        train_model(train_loader, val_loader, model, NUM_EPOCHS, start_epoch, best_accuracy)
        
    except ValueError as e:
        print(f"\n[資料錯誤] {e}\n請檢查 {DATA_DIR} 目錄，或確認下載是否成功。")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n[一般錯誤] 訓練過程中發生錯誤: {e}")