import numpy as np
from sklearn import datasets
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.layers import Conv2D, Flatten, MaxPooling2D, Dropout
from sklearn.metrics import r2_score, accuracy_score
import time


#1. 데이터
datasets = load_diabetes()
x, y = datasets.data, datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=72)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)  # (309, 10) (133, 10) (309,) (133,)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(309, 2, 5, 1) 
x_test = x_test.reshape(133, 2, 5, 1)
print(x_train.shape)    
print(np.unique(x_train, return_counts=True))


#2. 모델 구성
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(1, 1), padding='same', 
                 activation='relu', input_shape=(2, 5, 1)))
model.add(Dropout(0.2))     
model.add(Conv2D(64, (1, 1), padding='same', activation='relu'))                
model.add(Dropout(0.25))     
model.add(Conv2D(64, (1, 1), padding='same', activation='relu'))
model.add(Dropout(0.3))     
model.add(Conv2D(64, (1, 1), padding='same', activation='relu'))   
model.add(Dropout(0.25))                 
model.add(Conv2D(64, (1, 1), padding='same', activation='relu'))                
model.add(Dropout(0.2))   
model.add(Conv2D(64, (1, 1), padding='same', activation='relu'))                
model.add(Dropout(0.2))   

model.add(Flatten())   
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='linear'))
model.summary()


#3. 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mse'])  
earlyStopping = EarlyStopping(monitor = 'val_loss', patience=100, mode='min', 
                              verbose=1, 
                              restore_best_weights=True)
start_time = time.time() 
hist = model.fit(x_train, y_train, epochs=100, batch_size=32,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)
end_time = time.time() - start_time


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])

y_predict = model.predict(x_test)  
r2 = r2_score(y_test, y_predict)
print('r2 스코어: ', r2)  


#================================= loss, accuracy ===================================#
# loss :  3952.070068359375
# r2 스코어:  0.3403177143852616
#=================================================================================#
