from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import numpy as np
import time


#1. 데이터

datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.7, random_state = 66)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)
print(np.min(x_train)) #0.0 
print(np.max(x_train)) #1.0

#2. 모델구성
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input

input1 = Input(shape=(13,))
dense1 = Dense(5)(input1)
dense2 = Dense(100, activation = 'relu')(dense1)
dense3 = Dense(100, activation = 'relu')(dense2)
dense4 = Dense(50, activation = 'relu')(dense3)
dense5= Dense(50, activation = 'relu')(dense4)
dense6 = Dense(50, activation = 'relu')(dense5)
dense7 = Dense(70, activation = 'relu')(dense6)
output1 = Dense(1)(dense7)

model = Model(inputs=input1, outputs=output1)

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=300, mode='auto', verbose=1, 
                              restore_best_weights=True)        

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=500, batch_size=5,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)

end_time = time.time() 

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)



print("걸린시간 : ", end_time)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('loss : ' , loss)
print('r2스코어 : ', r2)


# #결과
# 걸린시간 :  1657158850.2760794
# loss :  [15.886625289916992, 2.8844499588012695]
# r2스코어 :  0.8077077885941848
