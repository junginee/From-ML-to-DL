

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import tensorflow as tf
from sklearn. datasets import fetch_covtype 
from sklearn.model_selection import train_test_split 

from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense 

from sklearn.metrics import accuracy_score 
import time

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y= datasets.target

print(datasets.feature_names)
print(datasets.DESCR)


print(x.shape, y.shape) #(581012, 54) (581012,)
print(np.unique(y, return_counts = True)) 
# y :[1 2 3 4 5 6 7]  / return_counts :[211840, 283301,  35754,   2747,   9493,  17367,  20510]

import pandas as pd
y = pd.get_dummies(y)
print(y)

x_train, x_test, y_train, y_test = train_test_split( x, y, train_size = 0.8, shuffle=True, random_state=68 )


scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)
print(np.min(x_train)) #0.0 
print(np.max(x_train)) #1.0

#2. 모델구성

model = Sequential()
model.add(Dense(5,input_dim = 54))
model.add(Dense(10, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(7, activation='softmax'))  # Classes 7


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=300, mode='auto', verbose=1, 
                              restore_best_weights=True)        

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=1, batch_size=10,
                 validation_split=0.2,
                 callbacks=[earlyStopping],verbose=1)

end_time = time.time() 

model.save_weights("./_save/keras23_14_save_fetch_covtype.h5")

#4. 평가, 예측

loss, acc = model.evaluate(x_test, y_test)
print('loss : ' , loss)
print('accuracy : ', acc) 

print("걸린시간 : ", end_time)

from sklearn.metrics import r2_score, accuracy_score  
y_predict = model.predict(x_test)
y_predict = tf.argmax(y_predict, axis= 1)
print(y_predict)

y_test = tf.argmax(y_test, axis= 1)
print(y_test)

acc= accuracy_score(y_test, y_predict)
print('acc스코어 : ', acc) 


