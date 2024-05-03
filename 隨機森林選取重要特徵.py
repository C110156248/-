import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder
#讀取資料
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
df = pd.read_csv('C:/Users/user/anaconda3/envs/myenv1/作業/housing.csv', header=None, encoding='utf-8')
#取得特徵名稱
feature_names = df.iloc[0, :9].tolist()
#Y
y = df.iloc[1:1000, 9].values
# y = y.astype(np.float64)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
#X
X = df.iloc[1:1000,:9].values
X = X.astype(np.float64)
# 移除有缺失值的資料(訓練)
nan_indices = np.isnan(X).any(axis=1)
X_cleaned = X[~nan_indices]
y_cleaned = y[~nan_indices]
# 建立隨機森林模型並訓練
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_cleaned, y_cleaned)

# 取得特徵重要性
feature_importances = rf_model.feature_importances_

#作圖
plt.figure(figsize=(8, 6))
plt.barh(feature_names, feature_importances, align='center', color='skyblue')
plt.xlabel('重要度')
plt.title('重要特徵')
plt.show()

