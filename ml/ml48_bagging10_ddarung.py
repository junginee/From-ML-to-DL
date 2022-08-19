import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler, StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

#1. 데이터
path = 'D:\study\_data\ddarung/'
train_set = pd.read_csv(path+'train.csv',index_col=0) # index_col = n : n번째 칼럼을 인덱스로 인식
# print(train_set)
# print(train_set.shape) # (1459, 10)

test_set = pd.read_csv(path+'test.csv', index_col=0)
# print(test_set)
# print(test_set.shape) # (715, 9)

# 결측치 중간값으로
print(train_set.info())
print(train_set.isnull().sum()) # 결측치 전부 더함
median = train_set.median()
train_set = train_set.fillna(median) # 결측치 중간값으로
print(train_set.isnull().sum()) # 없어졌는지 재확인

x = train_set.drop(['count'], axis=1) # axis = 0은 열방향으로 쭉 한줄(가로로 쭉), 1은 행방향으로 쭉 한줄(세로로 쭉)
y = train_set['count']

print(x.shape, y.shape) # (1328, 9) (1328,)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=66, train_size=0.8, shuffle=True)

#2. 모델
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVC

scaler = StandardScaler()
# scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


model = LinearSVC(), LinearRegression(),DecisionTreeRegressor(),RandomForestRegressor()

for i in model :
    model = i
    new_model = BaggingRegressor(i,
                          n_estimators=100, 
                          n_jobs=-1,
                          random_state=68
                          )
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    print(i,'모델 score :',round(r2_score(y_test, y_predict),4),'\n') 
    
# LinearSVC() 모델 score : 0.2179 

# LinearRegression() 모델 score : 0.5436 

# DecisionTreeRegressor() 모델 score : 0.5623

# RandomForestRegressor() 모델 score : 0.7678     