import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from point_to_picture import keypoints_to_image
import mediapipe as mp

# 載入模型
model = load_model('./cnn_test.keras')

# 初始化 Mediapipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# 定義類別名稱
class_names = ['a1', 'a2', 'b1','b2','c1']  # 根據您的模型類別數進行修改

# 開啟攝像頭
cap = cv2.VideoCapture(0)

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            keypoints = []
            for landmark in results.pose_landmarks.landmark:
                x = landmark.x * frame.shape[1]
                y = landmark.y * frame.shape[0]
                z = landmark.z
                keypoints.append([x, y, z])

            keypoints = np.array(keypoints)
            # 將關鍵點轉換為圖像
            keypoints_image = keypoints_to_image(keypoints)
            keypoints_image = np.expand_dims(keypoints_image, axis=0)  # 添加通道維度
            # 生成預測
            preds = model.predict(keypoints_image)
            if preds.shape[1] != len(class_names):
                raise ValueError(f"模型輸出類別數 ({preds.shape[1]}) 與 class_names 列表長度 ({len(class_names)}) 不匹配")

            class_index = np.argmax(preds[0])
            predicted_class = class_names[class_index]

            # 在影像上顯示預測結果
            cv2.putText(frame, f' {predicted_class}{preds[0][class_index]:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 顯示影像
        cv2.imshow('Camera', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
except Exception as e:
    print(f"An error occurred: {e}")

# 釋放資源
cap.release()
cv2.destroyAllWindows()