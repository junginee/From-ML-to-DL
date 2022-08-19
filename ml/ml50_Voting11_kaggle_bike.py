import numpy as np
import pandas as pd

from sklearn.ensemble import VotingClassifier,VotingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")

# 1. 데이터
path = 'D:\study\_data\kaggle_bike\\'
train_set = pd.read_csv(path+'train.csv')
# print(train_set)
# print(train_set.shape) # (10886, 11)

test_set = pd.read_csv(path+'test.csv')
# print(test_set)
# print(test_set.shape) # (6493, 8)

# datetime 열 내용을 각각 년월일시간날짜로 분리시켜 새 열들로 생성 후 원래 있던 datetime 열을 통째로 drop
train_set["hour"] = [t.hour for t in pd.DatetimeIndex(train_set.datetime)]
train_set["day"] = [t.dayofweek for t in pd.DatetimeIndex(train_set.datetime)]
train_set["month"] = [t.month for t in pd.DatetimeIndex(train_set.datetime)]
train_set['year'] = [t.year for t in pd.DatetimeIndex(train_set.datetime)]
train_set['year'] = train_set['year'].map({2011:0, 2012:1})

test_set["hour"] = [t.hour for t in pd.DatetimeIndex(test_set.datetime)]
test_set["day"] = [t.dayofweek for t in pd.DatetimeIndex(test_set.datetime)]
test_set["month"] = [t.month for t in pd.DatetimeIndex(test_set.datetime)]
test_set['year'] = [t.year for t in pd.DatetimeIndex(test_set.datetime)]
test_set['year'] = test_set['year'].map({2011:0, 2012:1})

train_set.drop('datetime',axis=1,inplace=True) # train_set에서 데이트타임 드랍
test_set.drop('datetime',axis=1,inplace=True) # test_set에서 데이트타임 드랍
train_set.drop('casual',axis=1,inplace=True) # casul 드랍 이유 모르겠음
train_set.drop('registered',axis=1,inplace=True) # registered 드랍 이유 모르겠음

#print(train_set.info())
# null값이 없으므로 결측치 삭제과정 생략

x = train_set.drop(['count'], axis=1)
y = train_set['count']
x = np.array(x)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, shuffle=True, train_size=0.8, random_state=1234)
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
    

# score : 0.9573
# XGBRegressor 정확도 : 0.9573
# LGBMRegressor 정확도 : 0.9573
# CatBoostRegressor 정확도 : 0.9573