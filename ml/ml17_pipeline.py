import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC, SVC
from sklearn. ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.decomposition import PCA

#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.8,shuffle=True, random_state=1234)

#2. 모델구성
model = Pipeline([('minmax', MinMaxScaler()), ('RF', RandomForestClassifier)])                                             
                                             
#3. 훈련
model.fit(x_train, y_train) 

#4. 평가, 예측
result = model.score(x_test, y_test)
print('[iris]')
print('model.score:', result)        

# [iris]
# model.score: 1.0                                     
