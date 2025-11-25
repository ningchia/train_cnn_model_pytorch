import pandas as pd
import os
import requests
from io import BytesIO
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import warnings
import time

# 忽略 PIL/Image 庫可能發出的警告
warnings.filterwarnings("ignore", category=UserWarning)

# --- 1. 配置與參數設定 ---
# 假設所有 CSV 檔案都在這個根目錄下
CSV_ROOT = "google-open-images-v6/csv" 
TRAIN_ROOT = "data_split"

# 我們要找的類別名稱及其目標資料夾
TARGET_CLASSES = {
    "Human hand": "hand-train",
    "Mug": "cup-train",
}

MAX_DOWNLOAD_PER_CLASS = 300
TARGET_SIZE = (224, 224)    # <--- 調整為 224x224 以跟MobileNetV2標準輸入尺寸一致

# 圖像處理轉換
resize_transform = transforms.Resize(TARGET_SIZE)

def download_and_process_images():
    # --- 步驟 1: 載入所有必要的 CSV 檔案 ---
    try:
        df_classes = pd.read_csv(os.path.join(CSV_ROOT, "class-descriptions-boxable.csv"))
        df_annotations = pd.read_csv(os.path.join(CSV_ROOT, "train", "simpler-oidv6-train-annotations-bbox.csv"))
        df_urls = pd.read_csv(os.path.join(CSV_ROOT, "train", "simpler-train-images-boxable-with-rotation.csv"))
    except FileNotFoundError as e:
        print(f"錯誤：找不到所需的 CSV 檔案，請檢查路徑設定 ({e.filename})。")
        return

    # --- 步驟 2: 獲取目標類別的 LabelID ---
    class_label_ids = {}
    for display_name, target_folder in TARGET_CLASSES.items():
        # 查找 DisplayName 對應的 LabelID
        label_id = df_classes[df_classes['LabelName'] == display_name]['LabelID']
        if not label_id.empty:
            class_label_ids[display_name] = label_id.iloc[0]
            print(f"找到類別 '{display_name}' 的 LabelID: {label_id.iloc[0]}")
        else:
            print(f"警告: 找不到類別 '{display_name}' 的 LabelID，跳過。")

    if not class_label_ids:
        print("沒有找到任何目標類別的 LabelID，程式結束。")
        return

    # --- 步驟 3: 篩選圖像 ID 和 URL ---
    all_target_image_info = {}
    for display_name, label_id in class_label_ids.items():
        # 篩選出包含該 LabelID 的所有 ImageID
        target_image_ids = df_annotations[df_annotations['LabelName'] == label_id]['ImageID'].unique()
        
        # 限制最多下載的數量
        target_image_ids = target_image_ids[:MAX_DOWNLOAD_PER_CLASS]
        
        # 將 ImageID 與 URL 數據合併，獲取下載連結
        image_info = df_urls[df_urls['ImageID'].isin(target_image_ids)]
        
        all_target_image_info[display_name] = image_info
        print(f"類別 '{display_name}' 找到 {len(image_info)} 張圖片URL。")


    # --- 步驟 4: 下載、縮放與儲存 (修正重試機制) ---
    print("\n--- 開始下載、縮放與儲存圖片 ---")
    DOWNLOAD_TIMEOUT = 15  # 放寬超時時間到 15 秒
    MAX_RETRIES = 3      # 最多重試 3 次

    for display_name, image_info in all_target_image_info.items():
        target_folder = TARGET_CLASSES[display_name]
        target_path = os.path.join(TRAIN_ROOT, target_folder)
        os.makedirs(target_path, exist_ok=True)
        
        download_count = 0
        
        # 這裡我們只對要處理的圖片數量進行進度條追蹤
        pbar = tqdm(image_info.iterrows(), total=len(image_info), desc=f"下載 {target_folder}", leave=True)
        
        for index, row in pbar:
            image_id = row['ImageID']
            url = row['OriginalURL']
            
            dest_file_path = os.path.join(target_path, f"{image_id}_{target_folder}_oid.jpg")

            # 檢查檔案是否已存在，避免重複下載
            if os.path.exists(dest_file_path):
                download_count += 1
                pbar.set_postfix({'status': 'Skipped (Exists)'})
                continue 

            # --- 新增重試機制 ---
            for attempt in range(MAX_RETRIES):
                try:
                    # 1. 下載
                    response = requests.get(url, timeout=DOWNLOAD_TIMEOUT)
                    response.raise_for_status() # 如果狀態碼不是 200, 會拋出 HTTPError
                    
                    # 2. 處理圖片
                    img = Image.open(BytesIO(response.content)).convert('RGB')
                    img_resized = resize_transform(img)
                    img_resized.save(dest_file_path, 'JPEG')
                    
                    # 成功後跳出重試迴圈
                    download_count += 1
                    pbar.set_postfix({'status': 'Success'})
                    break # 成功下載，進入下一個 image_id
                    
                except requests.exceptions.HTTPError as http_err:
                    # 4xx/5xx 錯誤 (如 404, 403)，通常是永久錯誤，不重試
                    # print(f"警告：下載 {image_id} 失敗 (HTTP 錯誤: {http_err.response.status_code})")
                    pbar.set_postfix({'status': f'HTTP Error {http_err.response.status_code}'})
                    break 
                    
                except requests.exceptions.RequestException as req_err:
                    # 處理連線錯誤 (超時、連線中斷等)
                    if attempt < MAX_RETRIES - 1:
                        # print(f"警告：下載 {image_id} 失敗 (重試 {attempt + 1}/{MAX_RETRIES})")
                        pbar.set_postfix({'status': f'Retry {attempt + 1}'})
                        time.sleep(1) # 暫停 1 秒後重試
                    else:
                        # 最後一次失敗
                        # print(f"警告：下載 {image_id} 最終失敗 (連線錯誤)")
                        pbar.set_postfix({'status': 'Connection Final Fail'})
                        
                except Exception as e:
                    # 處理圖片處理失敗（損壞的圖片檔案）
                    # print(f"警告：處理圖片 {image_id} 失敗 ({e})")
                    pbar.set_postfix({'status': 'Image Corrupt'})
                    break # 圖片損壞是永久性錯誤，無需重試
                    
            # --- 處理所有重試都失敗的情況 ---
            else:
                # 只有當 'break' 未被執行時（即所有重試都失敗），才會執行這裡
                continue

        print(f"✅ 類別 {display_name} 成功加入 {download_count} 張圖片到 {target_folder}。")
    
    print("\n--- Open Images 數據下載與合併完成 ---")
    print(f"所有檔案已儲存至 {TRAIN_ROOT} 中的對應資料夾，且尺寸為 {TARGET_SIZE}。")

if __name__ == '__main__':
    download_and_process_images()
