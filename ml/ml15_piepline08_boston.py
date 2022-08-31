import numpy as np
from sklearn.datasets import load_boston
from sklearn.svm import LinearSVC, SVC
from sklearn. ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import make_pipeline


#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.8,shuffle=True, random_state=1234)


#2. 모델구성
model = make_pipeline(MinMaxScaler(),RandomForestRegressor())  
                                          
                                             
#3. 훈련
model.fit(x_train, y_train) 


#4. 평가, 예측
result = model.score(x_test, y_test)
print('boston')
print('model.score:', round(result,4))    

# boston
# model.score: 0.9126                                       
