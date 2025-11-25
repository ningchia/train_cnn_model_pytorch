import torch
import torch.nn as nn
# --- 新增: 從 model_defs 模組導入模型結構 ---
from model_defs import CleanCNN

import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os
import time
from PIL import Image
import warnings

# 忽略 PIL/Image 庫可能發出的警告
warnings.filterwarnings("ignore", category=UserWarning)

# --- 1. 配置與參數設定 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "data_split"
MODEL_SAVE_PATH = "trained_model"
CHECKPOINT_FILE = "latest_checkpoint.pth"  # 統一檢查點檔案名稱
NUM_EPOCHS = 300 # 總 Epoch 數 (新的總數，從 start_epoch 開始累加)
BATCH_SIZE = 32
LEARNING_RATE = 0.001
TRANSFER_LEARNING_LR = 0.0001 # 遷移學習時，如果解凍，使用較低的 LR

# --- 3. 數據加載：CustomSplitDataset (已修正 nothing-train 擴增邏輯) ---
class CustomSplitDataset(Dataset):
    def __init__(self, root_dir, split_type, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = [] 
        self.classes = [] 
        
        all_subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        class_names = set()
        
        # --- 修正 nothing-train 擴增邏輯 (約原 77 行附近) ---
        target_dirs = []
        
        if split_type == 'train':
            # 處理 nothing 類別的優先級邏輯
            aug_nothing_dir = 'nothing-train-augmented'
            nothing_aug_path = os.path.join(root_dir, aug_nothing_dir)
            nothing_aug_exists = os.path.isdir(nothing_aug_path)
            
            # 暫存已經處理過的類別名稱，防止重複加載
            processed_classes = set()
            
            if nothing_aug_exists:
                # 優先使用 augmented 訓練集
                target_dirs.append(aug_nothing_dir)
                class_names.add('nothing')
                processed_classes.add('nothing')
            
            # 遍歷所有子目錄，處理其他類別 (hand, cup) 和作為 fallback 的 nothing-train
            suffix = f"-{split_type}" # 原來的後綴，用於匹配 hand-train 和 cup-train
            for d in all_subdirs:
                if d.endswith(suffix):
                    class_name = d.rsplit('-', 1)[0]
                    
                    if class_name == 'nothing':
                        if 'nothing' not in processed_classes:
                            # nothing-train: 只有在 nothing-train-augmented 不存在時才使用
                            target_dirs.append(d)
                            class_names.add(class_name)
                            processed_classes.add('nothing')
                    else:
                        # hand-train, cup-train 等其他類別，直接加入
                        target_dirs.append(d)
                        class_names.add(class_name)
                        processed_classes.add(class_name)
                        
        else:
            # 處理非 train 集合 (例如 validate)，使用原始的 -split_type 邏輯
            suffix = f"-{split_type}" 
            for d in all_subdirs:
                 if d.endswith(suffix):
                      target_dirs.append(d)
                      class_name = d.rsplit('-', 1)[0]
                      class_names.add(class_name)

        # --- 修正結束 ---
        
        self.classes = sorted(list(class_names))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        print(f"[{split_type.upper()}] 檢測到的類別: {self.classes}")
        
        # 遍歷選定的資料夾來加載樣本
        for d in target_dirs:
            # 確定當前資料夾對應的類別名稱和索引
            if d == 'nothing-train-augmented':
                 class_name = 'nothing'
            else:
                 # 假設格式是 class-split
                 class_name = d.rsplit('-', 1)[0]
                 
            if class_name in self.class_to_idx:
                class_idx = self.class_to_idx[class_name]
                folder_path = os.path.join(root_dir, d)
                
                for img_file in os.listdir(folder_path):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.samples.append((os.path.join(folder_path, img_file), class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_idx = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            print(f"警告: 無法讀取檔案 {img_path}")
            return self.__getitem__((idx + 1) % len(self)) 

        if self.transform:
            image = self.transform(image)
            
        return image, class_idx

# --- 數據轉換 (Transform)：針對 MobileNetV2 進行調整 ---
# 訓練集專用轉換 (包含數據擴增)
train_transform = transforms.Compose([
    # 統一將輸入影像縮放到 256x256 # <--- 調整為 224x224 以跟MobileNetV2標準輸入尺寸一致
    transforms.Resize((224, 224)), 
    # *** 關鍵新增：隨機水平翻轉 ***
    transforms.RandomHorizontalFlip(), 
    # ********************************
    transforms.ToTensor(), 
    # 使用 ImageNet 標準化的均值和標準差
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

# 驗證集/測試集專用轉換 (不包含隨機擴增)
val_transform = transforms.Compose([
    # 統一將輸入影像縮放到 256x256 # <--- 調整為 224x224 以跟MobileNetV2標準輸入尺寸一致
    transforms.Resize((224, 224)), 
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

def get_loaders(data_dir, batch_size):
    # 這裡無需修改 (原第 113 行)，因為 CustomSplitDataset 已處理優先級邏輯
    train_dataset = CustomSplitDataset(data_dir, 'train', train_transform)
    val_dataset = CustomSplitDataset(data_dir, 'validate', val_transform)
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
         raise ValueError(f"訓練集或驗證集為空。訓練集: {len(train_dataset)}, 驗證集: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, len(train_dataset.classes)

def calculate_accuracy(loader, model):
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

# --- 新增功能函式 (保持不變) ---

def save_checkpoint(epoch, model, optimizer, best_acc, path):
    """保存模型、優化器狀態、epoch 和最佳準確度"""
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
    """載入檢查點並返回起始 epoch 和最佳準確度"""
    if not os.path.exists(path):
        return 0, 0.0, False # start_epoch, best_accuracy, is_resumed

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


def freeze_first_layers(model):
    """
    凍結模型的第一組 Conv2d (索引 0) 和 BatchNorm2d (索引 1)。
    注意這只是做實驗. 只有遷移式學習才要用這個. 
    """
    # 凍結 Conv2d (index 0)
    for param in model.model[0].parameters():
        param.requires_grad = False
        
    # 凍結 BatchNorm2d (index 1)
    for param in model.model[1].parameters():
        param.requires_grad = False
        
    print("\n*** [遷移學習] 已凍結第一層 Conv2d (索引 0) 和 BatchNorm2d (索引 1) 的參數。 ***")


# --- 4. 訓練流程主函式 (已修正儲存邏輯與錯誤捕獲) ---
def train_model(train_loader, val_loader, model, total_epochs, start_epoch, initial_best_acc):
    
    # 優化器只追蹤 requires_grad=True 的參數
    # 注意這只是做實驗. 只有遷移式學習才要用這個. 
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=LEARNING_RATE if start_epoch == 0 else TRANSFER_LEARNING_LR
    )
    
    criterion = nn.CrossEntropyLoss()
    
    best_accuracy = initial_best_acc # 從載入的歷史最高準確度開始
    visual_best_acc = best_accuracy
    
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    checkpoint_path = os.path.join(MODEL_SAVE_PATH, CHECKPOINT_FILE)
    
    # --- 儲存邏輯的起始 Epoch ---
    # 儲存功能將在 '總目標 Epoch - 10' 時啟動。
    saving_start_epoch = total_epochs - 10 

    # 如果是從檢查點恢復訓練，需要重新設定學習率
    if start_epoch > 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = TRANSFER_LEARNING_LR if any(p.requires_grad for p in model.parameters()) else LEARNING_RATE
        print(f"[續訓] 設置當前學習率為 {optimizer.param_groups[0]['lr']}")
    
    print(f"\n--- 開始訓練 (總目標 Epoch: {total_epochs}, 從 Epoch {start_epoch + 1} 開始) ---")
    print(f"注意: 模型儲存功能將在第 {saving_start_epoch + 1} 個 Epoch 啟動。")
    print("      可隨時按下 Ctrl+C 提前結束訓練，並儲存當前最佳模型。")
    
    try:
        for epoch in range(start_epoch, total_epochs):
            current_epoch_num = epoch + 1
            
            # --- 訓練階段 ---
            model.train() 
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
            
            # 1. 更新全局顯示的最佳準確度
            if val_accuracy > visual_best_acc:
                visual_best_acc = val_accuracy
                print_message += f" (新歷史最高: {visual_best_acc:.2f}%)"
            
            # 2. 判斷是否在儲存區間 (倒數 10 個 Epoch)
            is_saving_epoch = current_epoch_num >= saving_start_epoch
            
            if is_saving_epoch:
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    # 儲存最佳模型權重和所有狀態
                    save_checkpoint(epoch, model, optimizer, best_accuracy, checkpoint_path)
                    print_message += f" -> **模型狀態已更新儲存** (目前最佳 Acc: {best_accuracy:.2f}%)"
                else:
                    print_message += f" (儲存區間內，目前最佳: {best_accuracy:.2f}%)"
            
            else:
                print_message += f" (儲存功能關閉，剩餘 {saving_start_epoch - current_epoch_num} 個 Epoch 啟動)"

            print(print_message)

    except KeyboardInterrupt:
        print("\n\n*** [使用者中斷] 偵測到 Ctrl+C，提前結束訓練。 ***")
        # 中斷時，我們儲存當前的模型狀態和訓練進度作為檢查點
        # 注意：此時的 epoch 變數是最後一個完整執行的 epoch
        save_checkpoint(epoch, model, optimizer, best_accuracy, checkpoint_path)

    except RuntimeError as e:
        # 捕獲 DataLoader 拋出的 RuntimeError
        if "DataLoader worker" in str(e):
             print("\n\n*** [DataLoader中斷] 偵測到 DataLoader worker 異常退出 (可能由 Ctrl+C 引起)。 ***")
             # 執行儲存邏輯
             try:
                 # 這裡我們使用上一個完整執行的 epoch 狀態來儲存檢查點
                 save_checkpoint(epoch, model, optimizer, best_accuracy, checkpoint_path)
                 print(f"-> 成功儲存檢查點，以防數據丟失。")
             except NameError:
                 # 如果在第一個 batch/epoch 之前就中斷，epoch 可能還沒有被定義
                 print("-> 無法儲存，因為異常發生在第一個 Epoch 開始之前。")
             except Exception as save_err:
                 print(f"-> 儲存檢查點時發生錯誤: {save_err}")
        else:
            # 如果是其他的 RuntimeError，重新拋出
            raise e
            
    finally:
        # 不論是正常結束還是中斷，都會執行這段
        print("-" * 50)
        print(f"訓練流程結束。")
        print(f"整體訓練過程中的最高準確度: {visual_best_acc:.2f}%\n")
        if best_accuracy > 0.0:
            print(f"最終儲存的最佳準確度: {best_accuracy:.2f}%")
            
# --- 5. 執行區塊 (保持不變) ---
if __name__ == '__main__':
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    checkpoint_path = os.path.join(MODEL_SAVE_PATH, CHECKPOINT_FILE)

    try:
        train_loader, val_loader, num_classes_detected = get_loaders(DATA_DIR, BATCH_SIZE)
        
        print(f"總訓練樣本數: {len(train_loader.dataset)}")
        print(f"總驗證樣本數: {len(val_loader.dataset)}")
        print(f"偵測到類別數量: {num_classes_detected}")
        
        # 步驟 1: 初始化模型和優化器 (先用預設參數)
        model = CleanCNN(num_classes=num_classes_detected).to(device)
        initial_optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) # 用來傳給載入函式

        # 步驟 2: 載入檢查點
        start_epoch, best_accuracy, is_resumed = load_checkpoint(checkpoint_path, model, initial_optimizer)
        
        # 步驟 3: 判斷是否為遷移式學習（從已有模型開始）
        if is_resumed and start_epoch > 0:
            freeze_first_layers(model) # 凍結第一層 Conv 和 BN
            print("\n*** 進入遷移式學習/斷點續訓模式 ***")

        # 步驟 4: 開始訓練 (將載入的狀態傳遞給訓練函式)
        train_model(train_loader, val_loader, model, NUM_EPOCHS, start_epoch, best_accuracy)
        
    except ValueError as e:
        print(f"\n[資料錯誤] {e}\n請檢查 {DATA_DIR} 目錄下的檔案是否齊全且符合 `class-split` 命名格式。")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n[一般錯誤] 訓練過程中發生錯誤: {e}")
