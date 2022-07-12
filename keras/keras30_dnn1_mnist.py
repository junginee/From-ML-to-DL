import numpy as np
import pandas as pd
from keras.datasets import mnist
from tensorflow.python.keras. models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical



#1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()


#x 데이터 전처리
x_train = x_train.reshape(60000,-1)
x_test = x_test.reshape(10000, -1)

#y 데이터 전처리
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(np.unique(y_train, return_counts=True))  

#x, y shape 출력
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

#2. 모델구성
model = Sequential()
#model.add(Dense(64, input_shape= (28*28, )))
model.add(Dense(64, input_shape= (784, )))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(32, activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(10, activation='softmax'))

# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'] )              

# Fit
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',patience=20, mode='auto',
                   verbose=1, restore_best_weights=False)    

model.fit(x_train, y_train, epochs=16, batch_size=32, verbose=1,   
          validation_split=0.3, callbacks=[es])

#4. 평가, 예측

#평가
loss = model.evaluate(x_test, y_test)
print('loss : ' , loss[0])
print('accuracy : ' , loss[1])


from sklearn.metrics import  accuracy_score  
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis= 1) #열(axis= 1)에서 최댓값을 구한다.

from tensorflow.keras.utils import to_categorical 
y_predict = to_categorical(y_predict)


acc= accuracy_score(y_test, y_predict)
print('acc스코어 : ', acc)


# loss :  0.19209639728069305
# accuracy :  0.9501000046730042
# acc스코어 :  0.9501