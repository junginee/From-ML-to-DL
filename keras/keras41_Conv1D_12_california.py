import numpy as np
from sklearn import datasets
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Dropout, LSTM, Flatten, Conv1D
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score, accuracy_score
import time

      
#1. 데이터
datasets = fetch_california_housing()
x, y = datasets.data, datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=66)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (14447, 8) (6193, 8) (14447,) (6193,)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(14447, 4, 2) 
x_test = x_test.reshape(6193, 4, 2)
print(x_train.shape, x_test.shape)   #(14447, 4, 2) (6193, 4, 2)
print(np.unique(x_train, return_counts=True))


#2. 모델 구성
model = Sequential()
model.add(Conv1D(32,4,activation='relu', input_shape=(4, 2)))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))     
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))     
model.add(Dense(32, activation='relu'))   
model.add(Dropout(0.3))                 
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='linear'))
model.summary()


#3. 훈련
model.compile(loss='mse', optimizer='adam',
              metrics=['mse'])   


earlyStopping = EarlyStopping(monitor = 'val_loss', patience=50, mode='min', verbose=1, 
                              restore_best_weights=True)



hist = model.fit(x_train, y_train, epochs=100, batch_size=100, 
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)



#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])      

y_predict = model.predict(x_test)  
r2 = r2_score(y_test, y_predict)
print('r2 스코어: ', r2)  


#================================= [LSTM]loss, accuracy ===========================#
# loss :  0.44939738512039185
# r2 스코어:  0.6724915053476774
#=================================================================================#


#================================= [Conv1D]loss, accuracy ===========================#
# loss :  0.4583950638771057
# r2 스코어:  0.6659342411732125
#=================================================================================#
