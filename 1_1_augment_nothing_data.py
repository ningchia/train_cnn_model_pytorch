import os
from torchvision import transforms
from PIL import Image
import warnings
from tqdm import tqdm
import shutil

# --- 1. 配置與參數設定 ---
BASE_DATA_DIR = "data_split"
SOURCE_CLASS_FOLDER = "nothing-train"  # <--- 修正為 nothing-train
TARGET_CLASS_FOLDER = "nothing-train-augmented"  # <--- 修正為 nothing-train-augmented
NUM_AUGMENTATIONS_PER_IMAGE = 4  # 每張圖片額外生成 4 個版本，總計擴增 5 倍
TARGET_SIZE = (256, 256) # 統一設定輸出尺寸，以匹配模型需求

# 忽略 PIL/Image 庫可能發出的警告
warnings.filterwarnings("ignore", category=UserWarning)

# --- 2. 定義積極的數據增廣 (Data Augmentation) 策略 ---
# 這些轉換會隨機應用，以產生多樣性
aggressive_augmentations = transforms.Compose([
    # 1. 顏色抖動 (Color Jitter): 隨機改變亮度、對比度、飽和度
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    
    # 2. 隨機裁切 (Random Crop) 和 重新調整大小 (Resize)
    # 這裡我們隨機裁切 85% 到 100% 的區域，然後縮放回 256x256
    transforms.RandomResizedCrop(
        size=TARGET_SIZE, 
        scale=(0.85, 1.0), # 隨機裁切的比例範圍
        ratio=(0.75, 1.3333) # 隨機長寬比
    ),
    
    # 3. 隨機翻轉 (Random Flip)
    transforms.RandomHorizontalFlip(p=0.5),
    
    # 4. 隨機旋轉 (Random Rotation)
    transforms.RandomRotation(degrees=15),
    
    # 5. 隨機高斯模糊 (Random Gaussian Blur)
    # 只有 30% 的機率應用模糊，模擬 WebCam 影像質量變化
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.3),
])

def augment_data():
    source_dir = os.path.join(BASE_DATA_DIR, SOURCE_CLASS_FOLDER)
    target_dir = os.path.join(BASE_DATA_DIR, TARGET_CLASS_FOLDER)

    print(f"來源資料夾: {source_dir}")
    print(f"目標資料夾: {target_dir}")
    
    if not os.path.isdir(source_dir):
        print(f"錯誤：找不到來源資料夾 {source_dir}。請檢查路徑。")
        return

    # 清理並創建目標資料夾
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
        print(f"已清除舊的目標資料夾 {target_dir}")
    os.makedirs(target_dir)

    image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    original_count = len(image_files)
    total_new_count = 0
    
    if original_count == 0:
        print(f"來源資料夾 {source_dir} 中沒有找到任何圖片。")
        return

    pbar = tqdm(image_files, desc="正在增廣圖片")
    
    for filename in pbar:
        img_path = os.path.join(source_dir, filename)
        
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"警告：無法讀取或轉換檔案 {filename} ({e})，跳過。")
            continue

        base_name, ext = os.path.splitext(filename)
        
        # 1. 儲存原始圖片 (作為第一個樣本)
        original_output_path = os.path.join(target_dir, filename)
        img.save(original_output_path)
        total_new_count += 1
        
        # 2. 生成並儲存增廣圖片
        for i in range(NUM_AUGMENTATIONS_PER_IMAGE):
            # 應用隨機增廣
            augmented_img = aggressive_augmentations(img)
            
            # 儲存新的檔案
            new_filename = f"{base_name}_aug_{i}{ext}"
            new_output_path = os.path.join(target_dir, new_filename)
            augmented_img.save(new_output_path)
            total_new_count += 1
            
    print("-" * 50)
    print(f"✅ 增廣完成！")
    print(f"原始圖片數量: {original_count} 張")
    print(f"擴增後總圖片數量: {total_new_count} 張 (原始數量的 {total_new_count / original_count:.1f} 倍)")
    print(f"所有檔案已儲存至: {target_dir}")

if __name__ == '__main__':
    augment_data()
