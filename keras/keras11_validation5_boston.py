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
model.fit(x_train, y_train, epochs = 100, batch_size = 1, validation_split=0.3)

# 4. 평가, 예측    
loss = model.evaluate(x_test, y_test)
print('loss : ',loss)

y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

#[keras08_1_boston]                                   
#loss :  23.53175163269043
#r2스코어 :  0.6978793840996296 

#[keras11_validation5_boston]                                                        
#loss :  28.776151657104492
#r2스코어 :  0.5520111442839735              

# validation dataset에 대한 성능은 학습을 중단하는 시점을 결정하기 위해 이용되고, test dataset에 대한 성능은 모델의 최종 정확도를 평가하기 위해 이용                                     
