from random import randrange
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA 
#https://www.youtube.com/watch?v=FgakZw6K1QQ

import sklearn as sk
print(sk.__version__) #0.24.2

import warnings
warnings.filterwarnings(action = 'ignore')

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target
# print(x.shape, y.shape) #(506, 13) (506,)

pca = PCA(n_components=11)  
x = pca.fit_transform(x)
# print(x.shape) #(506, 2)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8, random_state=123, shuffle=True
)

#2. 모델
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBClassifier, XGBRegressor

model = RandomForestRegressor()

#3. 훈련
model.fit(x_train, y_train) 

#4. 평가, 예측
results = model.score(x_test, y_test)
print('결과 :', round(results,4))

# PCA 사용 전
# 결과 : 0.7795

# PCA 사용 후(n_components=11)
# 결과 : 0.7967
