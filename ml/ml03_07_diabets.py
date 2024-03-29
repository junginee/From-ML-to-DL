import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler


datasets =load_diabetes()
x = datasets.data
y = datasets.target
x_train,x_test,y_train, y_test = train_test_split(x,y, train_size=0.75, shuffle=True, random_state=72)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)


#2. 모델구성
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LinearRegression 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn. ensemble import RandomForestRegressor

model = Perceptron(),LinearSVC(),LinearRegression(),KNeighborsRegressor(),DecisionTreeRegressor(),RandomForestRegressor()


for i in model:    
    model = i
    

    #3. 컴파일, 훈련
    model.fit(x_train, y_train)


    #4. 평가, 예측
    result = model.score(x_test,y_test)   
    y_predict = model.predict(x_test)    
    print(f"{i} : ", round(result,4))
    
# Perceptron() :  0.009
# LinearSVC() :  0.009
# LinearRegression() :  0.65
# KNeighborsRegressor() :  0.5608
# DecisionTreeRegressor() :  -0.0434
# RandomForestRegressor() :  0.5646
