import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

#1. 데이터
datasets =load_diabetes()
x,y = datasets.data, datasets.target

print(x.shape, y.shape) #(569, 30) (569,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=66, train_size=0.8, shuffle=True)

#2. 모델
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVC

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


model = LinearSVC(), LinearRegression(),DecisionTreeRegressor(),RandomForestRegressor()

for i in model :
    model = i
    new_model = BaggingRegressor(i,
                          n_estimators=100, 
                          n_jobs=-1,
                          random_state=77
                          )
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    print(i,'모델 score :',round(r2_score(y_test, y_predict),4),'\n') 
    
# LinearSVC() 모델 score : -0.1563 

# LinearRegression() 모델 score : 0.5064

# DecisionTreeRegressor() 모델 score : -0.1463

# RandomForestRegressor() 모델 score : 0.3756
