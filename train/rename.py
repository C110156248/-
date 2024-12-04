import os

# 定義資料集路徑
dataset_paths = ["C:/Users/user/Desktop/dnace_calssifier/A",
                "C:/Users/user/Desktop/dnace_calssifier/B",                
                "C:/Users/user/Desktop/dnace_calssifier/C",]


# 遍歷每個資料夾並重新命名圖片文件
for dataset_path in dataset_paths:
    file_list = os.listdir(dataset_path)
    image_count = 1
    for filename in file_list:
        if filename.endswith(".jpg") or filename.endswith(".png"):
            old_file_path = os.path.join(dataset_path, filename)
            new_file_name = f"image{image_count}.jpg"  # 這裡假設所有圖片都轉換為 .jpg 格式
            new_file_path = os.path.join(dataset_path, new_file_name)
            # 檢查目標檔案是否已存在
            if not os.path.exists(new_file_path):
                os.rename(old_file_path, new_file_path)
                image_count += 1
            else:
                print(f"檔案 {new_file_path} 已存在，跳過重新命名")

print("重新命名完成")