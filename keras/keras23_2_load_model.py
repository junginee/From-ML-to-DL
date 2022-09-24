import numpy as np 
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

#1.데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.7, random_state = 66)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)
print(np.min(x_train)) #0.0 
print(np.max(x_train)) #1.0


#2.모델구성
# model = Sequential()
# model.add(Dense(64, input_dim=13)) 
# model.add(Dense(32))
# model.add(Dense(16))
# model.add(Dense(8))
# model.add(Dense(1))
# model.summary()

# model.save("./_save/keras23_2_load_model.h5")

model = load_model("./_save/keras23_2_load_model.h5")
model.summary()


'''
#3. 컴파일, 훈련

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='loss', patience=200, mode='min', 
                              verbose=1,restore_best_weights=True)

model.compile(loss='mse', optimizer='adam')

import time    #시간
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=100,
                 validation_split=0.2,
                 callbacks = [earlyStopping],
                 verbose=1)

end_time = time.time() - start_time

#4. 평가, 예측

loss = model.evaluate(x_test, y_test)
print('loss : ' , loss)
print("걸린시간 : ", end_time)
y_predict = model.predict(x_test) 

#R2결정계수(성능평가지표)
r2 = r2_score(y_test, y_predict) 
print('r2스코어 : ' , r2) 

     
'''
