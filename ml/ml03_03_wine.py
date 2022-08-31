import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.datasets import load_wine 
from sklearn.model_selection import train_test_split 
import tensorflow as tf
tf.random.set_seed(66)

#1. 데이터
datasets = load_wine()
x = datasets.data
y= datasets.target

x_train, x_test, y_train, y_test = train_test_split( x, y, train_size = 0.8, shuffle=True, random_state=68 )

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)


#2. 모델구성
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron, LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn. ensemble import RandomForestClassifier

model = Perceptron(),SVC(),KNeighborsClassifier(),LogisticRegression(),DecisionTreeClassifier(),RandomForestClassifier()

for i in model:    
    model = i
    

    #3. 컴파일, 훈련
    model.fit(x_train, y_train)


    #4. 평가, 예측
    result = model.score(x_test,y_test)   
    y_predict = model.predict(x_test)
    print(f"{i} : ", round(result,4))
    
'''    
Perceptron() :  0.9167
SVC() :  1.0
KNeighborsClassifier() :  0.9444
LogisticRegression() :  0.9167
DecisionTreeClassifier() :  0.8889
RandomForestClassifier() :  0.9722    
'''
