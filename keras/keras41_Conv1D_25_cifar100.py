from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv1D, Flatten, MaxPooling2D, Dropout # 이미지 작업은 2D
from keras.datasets import mnist, cifar100
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler

#1. 데이터

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)   # (10000, 32, 32, 3) (10000, 1)

#--------------------------------------------------------------------
# x에 대한 전처리

scaler = MinMaxScaler() 
x_train = x_train.reshape(50000,-1)
x_test = x_test.reshape(10000,-1) 

print(x_train.shape, x_test.shape) #(50000, 3072) (10000, 3072)

scaler.fit(x_train)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


x_train = x_train.reshape(50000,96,32)
x_test = x_test.reshape(10000,96,32) 


print(x_train.shape, x_test.shape) #(50000, 96, 32) (10000, 96, 32)

#--------------------------------------------------------------------
# y에 대한 전처리(원핫인코딩)

import numpy as np
print(np.unique(y_train)) #0~99까지 총 100개

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)



#2. 모델구성
model = Sequential()
model.add(Conv1D(32, 10,activation='relu', input_shape=(96,32)))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='softmax'))
# model.summary()



#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint


earlyStopping =EarlyStopping(monitor='val_loss', patience=1, mode='min', verbose=1, 
                             restore_best_weights=True) 



hist = model.fit(x_train, y_train, epochs=100, batch_size=5000, 
                validation_split=0.2,
                callbacks=[earlyStopping], 
                verbose=1)



#4. 평가, 예측
loss = model.evaluate(x_test, y_test) 
print('loss : ', loss[0])
print('accuracy : ', loss[1])

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score, accuracy_score
y_predict = np.argmax(y_predict, axis= 1)
y_predict = to_categorical(y_predict)

#acc = accuracy_score(y_test, y_predict)
#print('acc스코어 : ', acc)

#[LSTM]
# loss :  4.607076644897461
# accuracy :  0.009999999776482582

# [Conv1D]
# loss :  3.907496452331543                                 
# accuracy :  0.10920000076293945
