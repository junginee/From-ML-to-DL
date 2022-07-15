from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Dropout, LSTM

from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import validation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)         # test도 카테고리컬해줘야 10으로 바뀜



print(x_train.shape, x_test.shape)   #(60000, 28, 28) (10000, 28, 28)
                      
                                          

print(np.unique(y_train, return_counts=True))   #10개 array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]



# 2. 모델구성

model  =  Sequential() 
model.add(LSTM(10, input_shape=(28, 28 ) ))
model.add(Dense(5, activation="relu") )
model.add(Dropout(0.2))
model.add(Dense(7,activation="relu") )
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

model.fit(x_train, y_train, epochs=100, batch_size=50, verbose=1,   
          validation_split=0.3, callbacks=[es])

# 4. 평가, 예측
# Evaluate
loss = model.evaluate(x_test, y_test)
print('loss : ', round(loss[0],4))             # 값이 2개가 나오는데 첫째로 로스가 나오고, 둘째로 accuracy가 나온다.
print('accuracy : ', round(loss[1],4))              # ★ accuracy빼고싶을때 loss[0]하면 리스트에서 첫번째만 출력하니까 로스만 찍을 수 있음★


# [LSTM]
# loss :  0.7812
# accuracy :  0.727
