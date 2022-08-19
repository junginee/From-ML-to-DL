import numpy as np
import pandas as pd
from sklearn.datasets import  fetch_covtype 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBClassifier, XGBRegressor 
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

#1. 데이터
datasets =  fetch_covtype ()
x = datasets.data
y = datasets.target

print(x.shape, y.shape) #(506, 13) (506,)


x_train, x_test, y_train, y_test = train_test_split(
    x,y, test_size = 0.2, random_state=123,
)

scaler = [StandardScaler(), MinMaxScaler(), MaxAbsScaler(), RobustScaler(), 
QuantileTransformer(),PowerTransformer(method='yeo-johnson'),] #PowerTransformer(method='box-cox')]

model = RandomForestClassifier()

for x in scaler :
  scaler = x         
  x_train = scaler.fit_transform(x_train)
  x_test = scaler.transform(x_test)
  model.fit(x_train, y_train)
  y_predict = model.predict(x_test)
  results = r2_score(y_test, y_predict)
  score = f1_score(y_test, y_predict, average = 'macro')
  class_name = scaler.__class__.__name__
  print('{0} 결과 : {1:.4f}'.format(scaler,results))
  print('{0} score결과 : {1:.4f}'.format(scaler,score))
  print() 