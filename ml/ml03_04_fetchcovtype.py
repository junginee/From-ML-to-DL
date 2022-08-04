import numpy as np
import tensorflow as tf
from sklearn. datasets import fetch_covtype 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from sklearn.svm import LinearSVC


#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y= datasets.target

print(datasets.feature_names)
print(datasets.DESCR)
print(x.shape, y.shape) #(581012, 54) (581012,)
print(np.unique(y, return_counts = True)) # y :[1 2 3 4 5 6 7]  / return_counts :[211840, 283301,  35754,   2747,   9493,  17367,  20510]

# import pandas as pd
# y = pd.get_dummies(y)
# print(y)

x_train, x_test, y_train, y_test = train_test_split( x, y, train_size = 0.8, shuffle=True, random_state=68 )


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