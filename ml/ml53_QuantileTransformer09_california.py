import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings("ignore")

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

print(x.shape, y.shape) #(506, 13) (506,)


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

# StandardScaler() 결과 : 0.8042
# MinMaxScaler() 결과 : 0.8062
# MaxAbsScaler() 결과 : 0.8061
# RobustScaler() 결과 : 0.8039
# QuantileTransformer() 결과 : 0.8070
# PowerTransformer() 결과 : 0.8043
