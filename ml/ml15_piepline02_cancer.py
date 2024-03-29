import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터

datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.8,shuffle=True, random_state=1234)

#2. 모델구성
from sklearn.svm import LinearSVC, SVC
from sklearn. ensemble import RandomForestClassifier

from sklearn.pipeline import make_pipeline

#model = SVC()
model = make_pipeline(MinMaxScaler(),RandomForestClassifier()) 
                                       
                                             
#3. 훈련
model.fit(x_train, y_train)  #piepline의  model.fit에서는 fit과 transform 동시 일어남

#4. 평가, 예측
result = model.score(x_test, y_test)
print('cancer')
print('model.score:', round(result,4)) 

# cancer
# model.score: 0.9298                                           
