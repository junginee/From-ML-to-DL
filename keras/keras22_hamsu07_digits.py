import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn. datasets import load_digits
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score

#1. 데이터
datasets = load_digits()
x = datasets.data
y= datasets.target

print(x.shape, y.shape) #(1797, 64) #(1797,)
print(np.unique(y)) #[0 1 2 3 4 5 6 7 8 9]

import tensorflow as tf
tf.random.set_seed(66)

from tensorflow.keras.utils import to_categorical 
y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split( x, y, train_size = 0.8, shuffle=True, random_state=68 )


scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)
print(np.min(x_train)) #0.0 
print(np.max(x_train)) #1.0

#2. 모델구성
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input

#함수형 모델
input1 = Input(shape=(64,))
dense1 = Dense(5)(input1)
dense2 = Dense(10, activation = 'relu')(dense1)
dense3 = Dense(25, activation = 'relu')(dense2)
dense4 = Dense(20, activation = 'relu')(dense3)
output1 = Dense(10,  activation='softmax')(dense4)

model = Model(inputs=input1, outputs=output1)

#Sequential 모델
# model = Sequential()
# model.add(Dense(5,input_dim = 64))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(25, activation='relu'))
# model.add(Dense(20, activation='relu'))
# model.add(Dense(10, activation='softmax'))  


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

import time
start_time = time.time()


from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=300, mode='auto', verbose=1, 
                              restore_best_weights=True)        


hist = model.fit(x_train, y_train, epochs=100, batch_size=10,validation_split=0.2,
                 callbacks=[earlyStopping],verbose=1)

end_time = time.time()

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)

print("걸린시간 : ", end_time)
print('loss : ' , loss)
print('accuracy : ', acc) 

from sklearn.metrics import r2_score, accuracy_score  
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis= 1)

y_test = np.argmax(y_test, axis= 1)
acc= accuracy_score(y_test, y_predict)
print('acc스코어 : ', acc) 

#[결과]
# 걸린시간 :  1657161465.3790865
# loss :  0.6763253808021545
# accuracy :  0.9222221970558167
# acc스코어 :  0.9222222222222223
