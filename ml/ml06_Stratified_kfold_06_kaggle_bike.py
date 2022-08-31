import numpy as np
import pandas as pd
from sqlalchemy import true
from sklearn.model_selection import train_test_split, KFold, cross_val_score,StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVR
from sklearn.linear_model import Perceptron, LinearRegression
import datetime as dt
import warnings
warnings.filterwarnings("ignore")

#1. 데이터
path = './_data/kaggle_bike/'
train_set = pd.read_csv(path + 'train.csv')             
test_set = pd.read_csv(path + 'test.csv')   

######## Year, month, day, time separation ############

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
print(np.min(x_train))  # 0.0
print(np.max(x_train))  # 1.0

print(np.min(x_test))  # 1.0
print(np.max(x_test))  # 1.0

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle = True, random_state=66)

#2. 모델구성
# model = LinearSVR()  
model = LinearRegression()               

#3,4. 컴파일, 훈련, 평가, 예측
# model.fit(x_train, y_train)
scores = cross_val_score(model,x,y,cv=kfold)
print('ACC : ',scores,'\ncross_val_score :', round(np.mean(scores),4))
