import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings("ignore")

#1. 데이터
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
    x,y, test_size = 0.2, random_state=1234,
)

scalers = [StandardScaler(), MinMaxScaler(), MaxAbsScaler(), RobustScaler(), QuantileTransformer(), PowerTransformer(method='yeo-johnson'),
           ] #PowerTransformer(method='box-cox')  
models = [LinearRegression(), RandomForestRegressor()] # , XGBRegressor

for i in scalers:
    scaler = i
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    for model in models:
        
            model = model
            model.fit(x_train, y_train)
            y_predict = model.predict(x_test)
            result = r2_score(y_test, y_predict)    
            print(model, scaler,'- 결과 :', round(result, 4))    
      
    print()

'''
LinearRegression() StandardScaler() - 결과 : 0.3907
RandomForestRegressor() StandardScaler() - 결과 : 0.9533

LinearRegression() MinMaxScaler() - 결과 : 0.3907
RandomForestRegressor() MinMaxScaler() - 결과 : 0.9542

LinearRegression() MaxAbsScaler() - 결과 : 0.3907
RandomForestRegressor() MaxAbsScaler() - 결과 : 0.9548

LinearRegression() RobustScaler() - 결과 : 0.3907
RandomForestRegressor() RobustScaler() - 결과 : 0.9535

LinearRegression() QuantileTransformer() - 결과 : 0.396
RandomForestRegressor() QuantileTransformer() - 결과 : 0.9543

LinearRegression() PowerTransformer() - 결과 : 0.4003
RandomForestRegressor() PowerTransformer() - 결과 : 0.9531

'''
