
import numpy as np
import tensorflow as tf
from sklearn. datasets import fetch_covtype #1. 데이터
from sklearn.model_selection import train_test_split #1. 데이터

from tensorflow.python.keras.models import Sequential #2. 모델구성
from tensorflow.python.keras.layers import Dense #2. 모델구성

from sklearn.metrics import accuracy_score #3,4  metrics로 accuracy 지표 사용


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


hist = model.fit(x_train, y_train, epochs=1, batch_size=10,
                 validation_split=0.2,
                 callbacks=[earlyStopping],verbose=1)



#4. 평가, 예측

#[loss, acc 출력방법 1]
loss, acc = model.evaluate(x_test, y_test)
print('loss : ' , loss)
print('accuracy : ', acc) 

#[loss, acc 출력방법 2]
results = model.evaluate(x_test, y_test)
print('loss : ' , results[0])
print('accuracy : ', results[1]) 

print("----------------------------------------")

from sklearn.metrics import r2_score, accuracy_score  
y_predict = model.predict(x_test)
y_predict = tf.argmax(y_predict, axis= 1)
print(y_predict)

y_test = tf.argmax(y_test, axis= 1)
print(y_test)

acc= accuracy_score(y_test, y_predict)
print('acc스코어 : ', acc) 

#[결과]




