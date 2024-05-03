import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder



plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
df = pd.read_csv('C:/Users/user/anaconda3/envs/myenv1/作業/housing.csv', header=None, encoding='utf-8')

#臨海距離
y = df.iloc[1:1000, 9].values
# y = y.astype(np.float64)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
#經緯度
X = df.iloc[1:1000].values
X=X[:, [col for col in range(X.shape[1]) if col != 9]]
X = X.astype(np.float64)


rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)


#作圖
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
plt.xlabel('經度')
plt.ylabel('緯度')
plt.title('隨機森林分類器')
plt.colorbar(label='臨海距離')
plt.show()


