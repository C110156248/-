import cv2
import mediapipe as mp
import os
import json 

# 定義資料集路徑
dataset_paths = ["C:/Users/user/Desktop/dnace_calssifier/A",
                "C:/Users/user/Desktop/dnace_calssifier/B",                
                "C:/Users/user/Desktop/dnace_calssifier/C",]

output_paths = ["C:/Users/user/Desktop/dnace_calssifier/json_output/A",
                "C:/Users/user/Desktop/dnace_calssifier/json_output/B",                
                "C:/Users/user/Desktop/dnace_calssifier/json_output/C",]

# 確保輸出資料夾存在
for output_path in output_paths:
    os.makedirs(output_path, exist_ok=True)
# 初始化 Mediapipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5)


# 檢視每張圖片
for dataset_path, output_path in zip(dataset_paths, output_paths):
    for filename in os.listdir(dataset_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(dataset_path, filename)
        
            # 讀取圖片
            image = cv2.imread(image_path)
            height, width, _ = image.shape
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
            # 使用 Mediapipe 進行關鍵點偵測
            results = pose.process(image_rgb)
    
            # 繪製關鍵點
            points = []
            if results.pose_landmarks:
                for landmark in results.pose_landmarks.landmark:
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    z = landmark.z
                    points.append([x, y, z])
                    cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
    
            # 儲存標註結果
            annotation = {
                "version": "1.0",
                "shapes": [
                    {
                        "label": "keypoint",
                        "points": points,
                        "group_id": None,
                        "shape_type": "point",
                        "flags": {}
                    }
                ],
                "imagePath": filename,
                "imageHeight": height,
                "imageWidth": width
            }
            
            annotation_path = os.path.join(output_path, filename.replace(".jpg", ".json").replace(".png", ".json"))
            with open(annotation_path, 'w') as f:
                json.dump(annotation, f, indent=4)
            
# 關閉所有 OpenCV 視窗
cv2.destroyAllWindows()