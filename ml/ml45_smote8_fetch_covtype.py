import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from imblearn.over_sampling import SMOTE
from sklearn.datasets import fetch_covtype
import pickle

x_train = pickle.load(open('D:\study_data\_save\_xg/covtype_smote_x_train.pkl', 'rb'))
y_train = pickle.load(open('D:\study_data\_save\_xg/covtype_smote_y_train.pkl', 'rb'))

# 1. 데이터

dataset = fetch_covtype()

x = dataset.data
y = dataset.target

print(x.shape) # (581012, 54)
print(y.shape) # (581012,)
print(np.unique(y, return_counts=True)) # [1 2 3 4 5 6 7] [211840, 283301,  35754,   2747,   9493,  17367,  20510]

_, x_test, _, y_test = train_test_split(x, y, train_size=0.8, random_state=66, shuffle=True, stratify=y)



scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression # 이진분류모델

model = RandomForestClassifier(random_state=12)

#3. 컴파일, 훈련

model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test)
print('model.score : ', result)
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print('acc : ', acc)
# print('f1_score(macro) : ', f1_score(y_test, y_pred, average='macro')) 
print('f1_score(micro) : ', f1_score(y_test, y_pred, average='micro'))
