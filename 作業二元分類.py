import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import LabelEncoder
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
df = pd.read_csv('./housing.csv', header=None, encoding='utf-8')
# print(df.tail())

#臨海距離
y = df.iloc[1:1000, 9].values
# y = y.astype(np.float64)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
#經緯度
X = df.iloc[1:1000, [0, 2]].values
X = X.astype(np.float64)

#二元分類
ppn = Perceptron(eta0=0.01, max_iter=1000)
ppn.fit(X, y)

#作圖

def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)

    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')

plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.scatter(x=X[:, 0], y=X[:, 1])
plt.show()
