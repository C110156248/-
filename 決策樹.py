import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.preprocessing import LabelEncoder
import pandas as pd
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
df = pd.read_csv('C:/Users/user/anaconda3/envs/myenv1/作業/housing.csv', header=None, encoding='utf-8')

#臨海距離
y = df.iloc[1:1000, 9].values
# y = y.astype(np.float64)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
#經緯度
X = df.iloc[1:1000, [0, 2]].values
X = X.astype(np.float64)

#決策樹
tree_model = DecisionTreeClassifier(criterion='gini', 
                                    max_depth=4, 
                                    random_state=1)
tree_model.fit(X, y)


class_names = label_encoder.classes_.tolist()
# print(class_names)

# 作圖
plt.figure(figsize=(12, 8))
plot_tree(tree_model, feature_names=df.columns[[0, 2]].tolist(), class_names=class_names, filled=True)

plt.show()