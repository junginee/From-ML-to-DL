from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout # 이미지 작업은 2D
from keras.datasets import mnist, cifar10
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler

#1. 데이터

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

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

x_train = x_train.reshape(50000, 32, 32, 3)  
x_test = x_test.reshape(10000, 32, 32, 3)

print(x_train.shape, x_test.shape) #(50000, 32, 32, 3) (10000, 32, 32, 3)

#--------------------------------------------------------------------
# y에 대한 전처리(원핫인코딩)

import numpy as np
print(np.unique(y_train)) #[0 1 2 3 4 5 6 7 8 9]

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)



#2. 모델구성
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(3,3), # 64 다음 레이어로 전달해주는 아웃풋 노드의 갯수, kernel size 이미지를 자르는 규격
                 padding='same', # 원래 shape를 그대로 유지하여 다음 레이어로 보내주고 싶을 때 주로 사용!
                 input_shape=(32, 32, 3))) # (batch_size, rows, columns, channels)   # 출력 : (N, 32, 32, 64)
model.add(MaxPooling2D()) # (N, 16, 16, 64)
model.add(Conv2D(32, (2,2), 
                 padding='valid', # 디폴트
                 activation='relu')) # filter = 32, kernel size = (2,2) # 출력 : (N, 15, 15, 32)
model.add(Conv2D(32, (2,2), 
                 padding='valid', # 디폴트
                 activation='relu')) # filter = 32, kernel size = (2,2) # 출력 : (N, 14, 14, 32)
model.add(Flatten()) # (N, 6272) 14*14*32
model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

from tensorflow.python.keras.callbacks import EarlyStopping



earlyStopping =EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1, 
                             restore_best_weights=True) 



hist = model.fit(x_train, y_train, epochs=1, batch_size=5, 
                validation_split=0.2,
                callbacks=[earlyStopping], # 최저값을 체크해 반환해줌
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

# loss :  1.3762474060058594
# accuracy :  0.5
