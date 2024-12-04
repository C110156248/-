import numpy as np
import os
import json
from sklearn.preprocessing import LabelEncoder

# 定義資料集路徑
annotation_paths =["C:/Users/user/Desktop/dnace_calssifier/json_output/A",
                   "C:/Users/user/Desktop/dnace_calssifier/json_output/B",                
                   "C:/Users/user/Desktop/dnace_calssifier/json_output/C",]

# 讀取標註資料
X = []
y = []

for annotation_path in annotation_paths:
    label = os.path.basename(annotation_path)  # 使用資料夾名稱作為標籤
    file_list = os.listdir(annotation_path)
    for filename in file_list:
        if filename.endswith(".json"):
            with open(os.path.join(annotation_path, filename), 'r') as f:
                annotation = json.load(f)
                points = annotation['shapes'][0]['points']
                X.append(points)
                y.append(label)  # 使用資料夾名稱作為標籤
                # print(label)

X = np.array(X)
y = np.array(y)

print(X.shape, y.shape)

# 將標籤轉換為數字
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
print(y)

# 儲存資料
np.save("X1.npy", X)
np.save("y1.npy", y)
