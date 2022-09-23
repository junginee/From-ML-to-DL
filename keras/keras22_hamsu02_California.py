from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import time


#1. 데이터
datasets = fetch_california_housing() 
x = datasets.data 
y = datasets.target 
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, shuffle=True, random_state=50)



scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)
print(np.min(x_train)) #0.0 
print(np.max(x_train)) #1.0

# print(x)
# print(y)
# print(x.shape, y.shape) #(20640, 8) (20640,)

# print(datasets.feature_names)
# print(datasets.DESCR)

#2. 모델구성
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input

#함수형 모델
input1 = Input(shape=(8,))
dense1 = Dense(5)(input1)
dense2 = Dense(50, activation = 'relu')(dense1)
dense3 = Dense(50, activation = 'relu')(dense2)
dense4 = Dense(10, activation = 'relu')(dense3)
dense5= Dense(20)(dense4)
dense6 = Dense(20)(dense5)
dense7 = Dense(10)(dense6)
dense8 = Dense(5, activation = 'relu')(dense7)
dense9 = Dense(10, activation = 'sigmoid')(dense8)
output1 = Dense(1)(dense9)

model = Model(inputs=input1, outputs=output1)

#Sequential 모델
# model = Sequential()
# model.add(Dense(5,input_dim=8))
# model.add(Dense(50, activation='relu'))
# model.add(Dense(50, activation='relu'))
# model.add(Dense(10, activation='relu'))
# model.add(Dense(20))
# model.add(Dense(20))
# model.add(Dense(10))
# model.add(Dense(5, activation='relu'))
# model.add(Dense(10, activation='sigmoid'))
# model.add(Dense(1))
             

#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1, 
                              restore_best_weights=True)        

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=250, batch_size=50,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)

end_time = time.time()

#4. 평가, 예측
print("걸린시간 : ", end_time)


loss = model.evaluate(x_test, y_test)
print('loss : ',loss)

 
y_predict = model.predict(x_test) 

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict) #R 제곱 = 예측값 (y_predict) / 실제값 (y_test) 
print('r2스코어 : ', r2)

#[결과]
# 걸린시간 :  1657159644.676034
# loss :  [0.30689719319343567, 0.3766382932662964]
# r2스코어 :  0.7692149729624157
