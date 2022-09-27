import numpy as np 
from sklearn.datasets import fetch_covtype
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,accuracy_score
from tensorflow.python.keras.callbacks import EarlyStopping
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler
import pandas as pd

#1.데이터 
datasets = fetch_covtype()
x= datasets.data
y= datasets.target

print(x.shape, y.shape) #(581012, 54) (581012, )
print(np.unique(y,return_counts=True)) #(array[1 2 3 4 5 6 7],array[211840, 283301,  35754,   2747,   9493,  17367,  20510] )


#사이킷런 원핫인코더
from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder(categories='auto',sparse= False)#False로 할 경우 넘파이 배열로 반환된다.
y = y.reshape(-1,1)
one.fit(y)
y = one.transform(y)

print(y)
print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3,shuffle=True,
                                                     random_state=58)

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test) 

#2.모델
model = Sequential()
model.add(Dense(40, input_dim=54))
model.add(Dense(30,activation ='relu'))
model.add(Dense(30,activation ='relu'))
model.add(Dense(30,activation ='relu'))
model.add(Dense(30,activation ='relu'))
model.add(Dense(7,activation ='softmax'))


#3.컴파일,훈련
model.compile(loss= 'categorical_crossentropy', optimizer ='adam', metrics='accuracy') 

earlyStopping= EarlyStopping(monitor='val_loss',patience=10,mode='min',restore_best_weights=True,verbose=1)

model.fit(x_train, y_train, epochs=100, batch_size=100,validation_split=0.2,callbacks=earlyStopping, verbose=1) 



#4.평가,예측
loss = model.evaluate(x_test,y_test)
print('loss: ', round(loss[0],4))


y_predict = model.predict(x_test) 

y_predict = y_predict.argmax(axis=1) 

y_test = y_test.argmax(axis=1) 
acc = accuracy_score(y_test,y_predict)
print('acc스코어: ', round(acc,4))
