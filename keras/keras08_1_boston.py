# [학습하기]

# from sklearn.datasets import load_boston #사이킷런의 datasets에 있는 load_boston을 가져온다.
# datasets = load_boston() #load_boston에 있는 값들은 datasets라는 변수에 저장한다.
# x = datasets.data #datasets에서 데이터는 x 변수에 저장한다. (x=피처, 컬럼, 열, 특성(방의 넓이, 크기, 흑인 등))
# y = datasets.target #datasets에서 타겟은 y 변수에 저장한다. (y=보스톤 집값 등)

# print(x) #x로 추출된 값은 데이터 값은 표준정규분포화가 이루어진 값들이다.
# print(y) 

# print(x.shape, y.shape) #(506, 13) (506,) #X는 13개의 열로 이루어졌으며 이에 따라 input_dim은 13이다. y는 506개의 스칼라와 1개의 백터로 이루어져 있다.

# print(datasets.feature_names) # ['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO' 'B' 'LSTAT'] 
#                               # datasets의 feature_names을 출력해라.
# print(datasets.DESCR) #이미지 참고


# [실습] 아래를 완성할 것    
# 1. train 0.7
# 2. R2 0.8 이상

# 1. 데이터
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston 

datasets = load_boston() 
x = datasets.data 
y = datasets.target 
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.70, shuffle=True, random_state=88)

# 2. 모델구성

model = Sequential()
model.add(Dense(5, input_dim=13))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(7))
model.add(Dense(1))


# 3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=350, batch_size=7)

# 4. 평가, 예측    
loss = model.evaluate(x_test, y_test)
print('loss : ',loss)

y_predict = model.predict(x_test) #왜 x_test가 들어갈까? x전체 데이터를 집어넣으면 과적합 걸릴 수 있기 때문에 훈련한 데이터(x_train)제외 후 훈련하지 않은 x_test로 설정한다.
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict) #R 제곱 =예측값 (y_predict) / 실제값 (y_test) 
print('r2스코어 : ', r2)
                                    
#loss :  23.53175163269043
#r2스코어 :  0.6978793840996296                                                         
                                                   