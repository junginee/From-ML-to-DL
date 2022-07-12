import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.datasets import fashion_mnist
from sklearn.preprocessing import MinMaxScaler

#1. 데이터
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()  # 이미지 데이터 불러오기, train, test구분

print(x_train.shape, y_train.shape) # (60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape) # (10000, 28, 28) (10000,)


#--------------------------------------------------------------------
# x에 대한 전처리
scaler = MinMaxScaler() 
x_train = x_train.reshape(60000, -1)  # scaler를 활용하기 위해서는 2차원 데이터로 변환해야함
x_test = x_test.reshape(10000, -1)

print(x_train.shape, y_train.shape) #(60000, 784) (60000,)

scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(60000, -1)  
x_test = x_test.reshape(10000, -1)

print(x_train.shape, y_train.shape) #(60000, 784) (60000,)

#--------------------------------------------------------------------
# y에 대한 전처리(원핫인코딩)
print(np.unique(y_train)) #[0 1 2 3 4 5 6 7 8 9]


y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)


#2. 모델구성

model = Sequential()
model.add(Dense(64, input_shape= (784, )))
model.add(Dropout(0.2))
model.add(Dense(16))
model.add(Dense(10))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

#3. 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', restore_best_weights=True)

model.fit(x_train, y_train, epochs=10, batch_size=2, validation_split=0.2, callbacks=[es])


#4. 예측, 평가



loss = model.evaluate(x_test, y_test)
print('loss : ' , loss[0])
print('accuracy : ', loss[1]) 

print("----------------------------------------")

from sklearn.metrics import  accuracy_score  
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis= 1) #열(axis= 1)에서 최댓값을 구한다.

from tensorflow.keras.utils import to_categorical 
y_predict = to_categorical(y_predict)


acc= accuracy_score(y_test, y_predict)
print('acc스코어 : ', acc)

# loss :  0.49629008769989014
# accuracy :  0.8292999863624573
# ----------------------------------------
# acc스코어 :  0.8293