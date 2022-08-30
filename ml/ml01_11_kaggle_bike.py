import numpy as np
import pandas as pd
from sqlalchemy import true
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVR
import datetime as dt

#1. 데이터
path = './_data/kaggle_bike/'
train_set = pd.read_csv(path + 'train.csv')
            
test_set = pd.read_csv(path + 'test.csv') 

######## 년, 월 ,일 ,시간 분리 ############

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

train_set.drop('datetime',axis=1,inplace=True) 
train_set.drop('casual',axis=1,inplace=True) 
train_set.drop('registered',axis=1,inplace=True)
test_set.drop('datetime',axis=1,inplace=True) 

x = train_set.drop(['count'], axis=1)  
print(x.shape) # (10886, 12)

y = train_set['count'] 
print(y.shape) # (10886,)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.75,
                                                    random_state=20
                                                    )

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)
test_set = scaler.transform(test_set)

#2. 모델구성
model = LinearSVR()                 

#3. 컴파일, 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test, y_test)
print("결과 : ", round(results,3)) 

y_predict = model.predict(x_test) 
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict) 
print('r2스코어 : ' , round(r2,3)) 

# 결과 :  0.32
# r2스코어 :  0.32
