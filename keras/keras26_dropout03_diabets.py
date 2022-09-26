import numpy as np
from sklearn import datasets
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, StandardScaler, RobustScaler
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score, accuracy_score
import time


#1. 데이터
datasets = load_diabetes()
x, y = datasets.data, datasets.target


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=66
)


#2. 모델 구성
model = Sequential()
model.add(Dense(100, activation='linear', input_dim=10))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.3))                                     # 연산 때 30% 비율로 랜덤하게 노드를 없앰
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))                                     # 연산 때 20% 비율로 랜덤하게 노드를 없앰
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='linear'))


#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])  

import datetime
date = datetime.datetime.now()      # 2022-07-07 17:21:42.275191
date = date.strftime("%m%d_%H%M")   # 0707_1723
print(date)

filepath = './_ModelCheckPoint/k26/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

earlyStopping = EarlyStopping(monitor = 'val_loss', patience=100, mode='min', 
                              verbose=1, 
                              restore_best_weights=True)
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, 
                      save_best_only=True, 
                      filepath="".join([filepath, '03_', date, '_', filename])
                      )

start_time = time.time() 
hist = model.fit(x_train, y_train, epochs=1000, batch_size=100,
                 validation_split=0.2,
                 callbacks=[earlyStopping, mcp],
                 verbose=1)
end_time = time.time() - start_time


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)  
r2 = r2_score(y_test, y_predict)
print('r2 스코어: ', r2)  


#================================= 1. 기본 출력 ===================================#
# loss :  [3216.153076171875, 3216.153076171875]
# r2 스코어:  0.4837942008323731
# _03_0707_1918_0028-2992.4802.hdf5
#=================================================================================#

#================================ 2. dorpout 적용 ================================#
# loss :  [3145.81201171875, 3145.81201171875]
# r2 스코어:  0.4950842050835256
# _03_0708_1056_0228-2951.7988.hdf5
#=================================================================================#
