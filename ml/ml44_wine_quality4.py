# 라벨 축소하기
# y라벨 값이 많다면? 정확도가 떨어질 수 있다. 이럴 때 라벨을 축소시켜 모델을 구성할 수 있다.

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import KFold,StratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV


#1.데이터
path = 'D:\study\_data\\'

datasets = pd.read_csv(path + 'winequality-white.csv', 
                   index_col=None, header=0, sep=';') #csv 파일은 통상 , or ; 형태로 되어 있음

print(datasets.shape) #(4898, 12)
print(datasets.head())
print(datasets.describe())
print(datasets.info())


print(type(datasets))  
# print(datasets2.shape)

x = datasets.to_numpy()[:, :11]
y = datasets.to_numpy()[:, 11]
print(x.shape, y.shape)

print(np.unique(y, return_counts = True))    
print(datasets['quality'].value_counts())
                         
print(y[:20])

newlist =[]

for i in y:
    if 3 <= i <= 5:
        newlist += [0]
    elif 7 <= i <= 9:
        newlist += [2]
    else :
        newlist += [1] 
     
y = np.array(newlist)
print(np.unique(newlist, return_counts=True)) # array([1640, 2198, 1060]
     
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.8, 
                                                    random_state=123, shuffle = True, stratify = y)

#2. 모델
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier( )

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
y_predict = model.predict(x_test)

score = model.score(x_test, y_test)

from sklearn.metrics import accuracy_score, f1_score
print('model.score :' , round(score,4))
print("accuracy score : ",
      round(accuracy_score(y_test, y_predict), 4 ))
print("f1_score(macro) : ",round(f1_score(y_test, y_predict, average = 'macro'),4))
print("f1_score(micro) : ",round(f1_score(y_test, y_predict, average = 'micro'),4))

# model.score : 0.7439
# accuracy score :  0.7439
# f1_score(macro) :  0.7415
# f1_score(micro) :  0.7439





