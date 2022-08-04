import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn. datasets import load_digits

#1. 데이터
datasets = load_digits()
x = datasets.data
y= datasets.target

print(x.shape, y.shape)
print(np.unique(y)) 

import tensorflow as tf
tf.random.set_seed(66)

x_train, x_test, y_train, y_test = train_test_split( x, y, train_size = 0.8, shuffle=True, random_state=68 )

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)
print(np.min(x_train)) 
print(np.max(x_train))


#2. 모델구성
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import Perceptron, LogisticRegression  #LogisicRegression 분류
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

# Perceptron() :  0.9528    
# SVC() :  0.975
# KNeighborsClassifier() :  0.9056
# LogisticRegression() :  0.9722
# DecisionTreeClassifier() :  0.8278
# RandomForestClassifier() :  0.975    