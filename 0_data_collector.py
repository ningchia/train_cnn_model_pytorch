import cv2
import os
import time
import shutil # 引入 shutil 確保 os.makedirs 的父目錄創建

RUN_IN_WSL = False  # 如果在 WSL 環境下運行，請設置為 True

# --- 1. 定義類別名稱和儲存目錄 ---
CLASS_NAMES = ["nothing", "hand", "cup"] 
DATASET_DIR = "dataset" # 根資料集目錄
IMAGE_SIZE = (320, 240) # 儲存影像的目標尺寸 (寬, 高)

# 建立按鍵(ASCII碼)到類別名稱的映射字典
CLASS_KEYS = {ord(str(i)): CLASS_NAMES[i] for i in range(len(CLASS_NAMES))}

# 用於儲存每個類別的圖片計數
class_counts = {}

# 確保資料集根目錄和所有類別目錄存在，並讀取現有計數
print("--- 初始化與現有檔案計數 ---")
for class_name in CLASS_NAMES:
    class_dir = os.path.join(DATASET_DIR, class_name)
    # 使用 exists_ok=True 和 parent=True 確保路徑層次都能被創建
    os.makedirs(class_dir, exist_ok=True) 
    
    # 計算該目錄下現有的 .jpg 檔案數量
    current_count = len([f for f in os.listdir(class_dir) if f.endswith('.jpg')])
    class_counts[class_name] = current_count
    
    print(f"目錄: {class_dir} | 現有圖片數量: {current_count}")

total_image_count = sum(class_counts.values())

# --- 2. 使用OpenCV從webcam拍照 ---
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("錯誤: 無法開啟網路攝影機。請檢查裝置連接或索引號。")
    exit()

# 關鍵設定 for WSL：將格式設為 MJPG (降低頻寬需求，增加相容性)
if RUN_IN_WSL:
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))


print("\n--- 資料收集程式啟動 ---")
key_info = [f"[{i}] 儲存: {CLASS_NAMES[i]}" for i in range(len(CLASS_NAMES))]
print(f"可收集類別: {', '.join(key_info)}")
print("按 [q] 結束程式。")


while True:
    # 讀取一幀畫面
    ret, raw_frame = cap.read() # <--- 這是乾淨的原始畫面

    if not ret:
        print("無法從網路攝影機接收畫面。正在退出...")
        break

    # 複製原始畫面，用於顯示輔助文字 (Display Frame)
    display_frame = raw_frame.copy() 
    
    # ----------------------------------------------------
    # 在畫面上顯示即時計數 (所有文字都畫在 display_frame 上)
    # ----------------------------------------------------
    y_offset = 30
    
    # 顯示總計數
    total_text = f"Total Collected: {total_image_count}"
    cv2.putText(display_frame, total_text, (10, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    
    # 顯示每個類別的計數
    for i, class_name in enumerate(CLASS_NAMES):
        count = class_counts[class_name]
        y_offset += 30
        count_text = f"[{i}] {class_name}: {count} samples"
        
        cv2.putText(display_frame, count_text, (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    # 顯示畫面
    cv2.imshow('Data Collector - Real-time Count', display_frame) # <--- 顯示帶文字的畫面

    # 捕獲使用者按鍵
    key = cv2.waitKey(1) & 0xFF

    # 4. 當使用者按下q時就結束執行
    if key == ord('q'):
        break
    
    # 3. 檢查按鍵是否在定義的類別按鍵中
    if key in CLASS_KEYS:
        class_name = CLASS_KEYS[key]
        class_dir = os.path.join(DATASET_DIR, class_name)

        # 調整影像大小
        # *** 關鍵變動: 使用 raw_frame 進行 resize 和儲存 ***
        resized_frame = cv2.resize(raw_frame, IMAGE_SIZE) 
        
        # 產生檔案名稱
        timestamp = time.time()
        filename_format = "%Y%m%d-%H%M%S"
        milliseconds = int((timestamp - int(timestamp)) * 1000)
        timestr = time.strftime(filename_format, time.localtime(timestamp))
        file_name = f"{timestr}-{milliseconds:03d}.jpg"
        
        save_path = os.path.join(class_dir, file_name)

        # 儲存影像
        cv2.imwrite(save_path, resized_frame)
        
        # 更新計數器
        class_counts[class_name] += 1
        total_image_count += 1
        
        print(f"[{class_name}] 成功收集第 {class_counts[class_name]} 張影像. 總計: {total_image_count}")
        
        # 在畫面上給予儲存成功的視覺回饋 (仍在 display_frame 的拷貝上繪製)
        feedback_frame = display_frame.copy()
        cv2.putText(feedback_frame, f"SAVED! ({class_name})", (10, y_offset + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
        cv2.imshow('Data Collector - Real-time Count', feedback_frame)
        cv2.waitKey(200)

# 釋放資源並關閉視窗
cap.release()
cv2.destroyAllWindows()
print("\n程式執行完畢。")
