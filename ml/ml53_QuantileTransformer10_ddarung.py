import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings("ignore")

#1. 데이터
path = 'D:\study\_data\ddarung\\'
train_set = pd.read_csv(path+'train.csv',index_col=0) 
# print(train_set)
# print(train_set.shape) # (1459, 10)

test_set = pd.read_csv(path+'test.csv', index_col=0)
# print(test_set)
# print(test_set.shape) # (715, 9)

# 결측치 중간값으로
print(train_set.info())
print(train_set.isnull().sum())
median = train_set.median()
train_set = train_set.fillna(median) 
print(train_set.isnull().sum())

x = train_set.drop(['count'], axis=1) 
y = train_set['count']

x_train, x_test, y_train, y_test = train_test_split(
    x,y, test_size = 0.2, random_state=1234,
)

scaler = [StandardScaler(), MinMaxScaler(), MaxAbsScaler(), RobustScaler(), 
QuantileTransformer(),PowerTransformer(method='yeo-johnson'),] #PowerTransformer(method='box-cox')]

model = RandomForestRegressor()

for x in scaler :
  scaler = x         
  x_train = scaler.fit_transform(x_train)
  x_test = scaler.transform(x_test)
  model.fit(x_train, y_train)
  y_predict = model.predict(x_test)
  results = r2_score(y_test, y_predict)
  class_name = scaler.__class__.__name__
  print('{0} 결과 : {1:.4f}'.format(scaler,results))
  print() 
  
'''
StandardScaler() 결과 : 0.7828

MinMaxScaler() 결과 : 0.7897

MaxAbsScaler() 결과 : 0.7927

RobustScaler() 결과 : 0.7864

QuantileTransformer() 결과 : 0.7789

PowerTransformer() 결과 : 0.7897

'''  
