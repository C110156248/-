from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
pd.set_option('display.max_columns', None)
# 讀取資料
data = pd.read_csv('C:/Users/user/anaconda3/envs/myenv1/作業/bank.csv',sep=';', encoding='utf-8')


#LabelEncoder
datalabel =['job','marital','education','default','housing','loan','contact','month','poutcome','y']
le = LabelEncoder()
for col in datalabel:
    data[col] = le.fit_transform(data[col])
X= data.drop(['y'], axis=1)
y= data['y']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立並訓練模型
model = SVC(max_iter=1000)
model.fit(X_train, y_train)

# 進行預測
y_pred = model.predict(X_test)

# 評估模型
conf_matrix = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# 印出結果
print("Confusion Matrix:")
print(conf_matrix)
print("Precision:", precision)
print("Accuracy:", accuracy)
print("Recall:", recall)
print("F1 Score:", f1)