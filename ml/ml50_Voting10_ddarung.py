import numpy as np
import pandas as pd

from sklearn.ensemble import VotingClassifier,VotingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import warnings
warnings.filterwarnings("ignore")

#1. 데이터
path = 'D:\study\_data\ddarung\\'
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

print(x.shape, y.shape) #


x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.7, random_state=1234)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from lightgbm import LGBMClassifier, LGBMRegressor

xg = XGBRegressor(learning_rate=0.05,
              reg_alpha=0.02,
              )  

lg = LGBMRegressor()
cat = CatBoostRegressor()

model = VotingRegressor(
    estimators=[('xg',xg),('lg',lg), ('cb',cat)],
    # voting = 'soft'      #hard => 분류모델 파라미터
)

#3. 훈련
model.fit(x_train, y_train)


#4. 평가, 예측
y_predict = model.predict(x_test)

score = r2_score(y_test, y_predict)
print('score :', round(score,4))


classfiers = [xg,lg,cat]
for model2 in classfiers:
    model2.fit(x_train, y_train, verbose = 0)
    y_predict = model2.predict(x_test)
    score2 = r2_score(y_test, y_predict)
    class_name = model2.__class__.__name__
    print('{0} 정확도 : {1:.4f}'.format(class_name,score))
    

# score : 0.8031
# XGBRegressor 정확도 : 0.8031
# LGBMRegressor 정확도 : 0.8031
# CatBoostRegressor 정확도 : 0.8031