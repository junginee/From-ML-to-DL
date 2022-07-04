import numpy as np
from sklearn.datasets import load_wine #1. 데이터
from sklearn.model_selection import train_test_split #1. 데이터

from tensorflow.python.keras.models import Sequential #2. 모델구성
from tensorflow.python.keras.layers import Dense #2. 모델구성

from sklearn.metrics import accuracy_score #3,4  metrics로 accuracy 지표 사용

#1. 데이터
datasets = load_wine()
x = datasets.data
y= datasets.target

print(x.shape, y.shape) #(178, 13) #(178,)
print(np.unique(y, return_counts = True)) #[0 1 2]

import tensorflow as tf
tf.random.set_seed(66)

from tensorflow.keras.utils import to_categorical 
y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split( x, y, train_size = 0.8, shuffle=True, random_state=68 )


#2. 모델구성

model = Sequential()
model.add(Dense(5,input_dim = 13))
model.add(Dense(10, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(3, activation='softmax'))  


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=300, mode='auto', verbose=1, 
                              restore_best_weights=True)        


hist = model.fit(x_train, y_train, epochs=100, batch_size=10,
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
y_predict = np.argmax(y_predict, axis= 1)
print(y_predict)

y_test = np.argmax(y_test, axis= 1)
print(y_test)

acc= accuracy_score(y_test, y_predict)
print('acc스코어 : ', acc) 

#[결과]
# loss :  0.22129157185554504
# accuracy :  0.8888888955116272
# ----------------------------------------
# [1 1 0 0 1 1 1 0 1 2 2 0 0 0 1 0 2 0 1 1 0 1 2 1 0 1 1 1 1 1 2 0 1 1 2 2]
# [1 1 0 0 1 1 1 0 0 2 2 0 0 0 1 0 2 0 1 1 0 1 2 1 0 1 1 1 1 2 2 0 2 1 2 1]
# acc스코어 :  0.8888888888888888

