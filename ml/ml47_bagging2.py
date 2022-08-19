import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#1. 데이터
datasets = load_breast_cancer()
x,y = datasets.data, datasets.target

print(x.shape, y.shape) #(569, 30) (569,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=66, train_size=0.8, shuffle=True, stratify=y)

#2. 모델
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


# LogisticRegression 모델을 wrapping 후 
# BaggingClassifier로 작동시킨다.
# Bagging 사용하고 싶은 모델 스케일링 처리!!


scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# model = BaggingClassifier(DecisionTreeClassifier(),
#                           n_estimators=100, 
#                           n_jobs=-1,
#                           random_state=123
#                           )

model = LogisticRegression(), DecisionTreeClassifier()

for i in model :
    model = i
    new_model = BaggingClassifier(i,
                          n_estimators=100, 
                          n_jobs=-1,
                          random_state=123
                          )
    model.fit(x_train, y_train)
    print(i,'모델 score :',round(model.score(x_test, y_test),4)) 


#3. 훈련
# model.fit(x_train, y_train)

# #4. 평가
# print(round(model.score(x_test, y_test),4)) #0.9825

#========================================================
# LogisticRegression() 모델 score : 0.9825
# DecisionTreeClassifier() 모델 score : 0.9211