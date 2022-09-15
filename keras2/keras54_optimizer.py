import numpy as np

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10]) 
y = np.array([1,3,5,4,7,6,7,11,9,7]) 

#2. 모델
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

model = Sequential()
model.add(Dense(1000,input_dim=1))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1))

#3. 컴파일, 훈련
import tensorflow as tf
from tensorflow.python.keras.optimizer_v2 import adam, adadelta, adagrad, adamax
from tensorflow.python.keras.optimizer_v2 import rmsprop,nadam
optilist = [adam.Adam,adadelta.Adadelta,adagrad.Adagrad,
        adagrad.Adagrad,adamax.Adamax,rmsprop.RMSprop,nadam.Nadam]
for i in optilist:
    learning_rate = 0.1
    optimizer = i(lr=learning_rate)
    model.compile(loss='mse',optimizer=optimizer)
    model.fit(x,y,epochs=50,batch_size=1,verbose=2)
    #4. 평가, 예측
    loss = model.evaluate(x,y)
    y_predict = model.predict([11])
    print('optimizer :',i.__name__,'Loss :',round(loss,4),'lr :',learning_rate, '예측 결과물 :',y_predict)
    
####################################################
# learning_rate = 0.1
# optimizer = adam.Adam(lr=learning_rate) 
# Loss : 3.0885 lr : 0.1 예측 결과물 : [[12.118365]]
####################################################
# learning_rate = 0.1
# optimizer = adadelta.Adadelta(lr=learning_rate)
# Loss : 2.3928 lr : 0.1 예측 결과물 : [[10.735626]]