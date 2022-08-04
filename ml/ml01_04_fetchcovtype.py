
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
model = LinearSVC()  

#3. 컴파일, 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
print("fetchcovtype")

results = model.score(x_test, y_test)
print("결과 acc : ", round(results,3)) 

from sklearn.metrics import r2_score, accuracy_score  
y_predict = model.predict(x_test)


acc= accuracy_score(y_test, y_predict)
print('acc스코어 : ', round(acc,3))  


