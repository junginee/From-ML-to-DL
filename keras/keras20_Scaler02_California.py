from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import time

datasets = fetch_california_housing() 
x = datasets.data 
y = datasets.target 
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, shuffle=True, random_state=50)


###############스캘러 방법#####################################
#scaler = StandardScaler()
#scaler = MinMaxScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
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
model = Sequential()
model.add(Dense(5,input_dim=8))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(5, activation='relu'))
model.add(Dense(10, activation='sigmoid'))
model.add(Dense(1))
                

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
loss = model.evaluate(x_test, y_test)
print('loss : ',loss)

print("걸린시간 : ", end_time)
y_predict = model.predict(x_test) 

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict) #R 제곱 = 예측값 (y_predict) / 실제값 (y_test) 
print('r2스코어 : ', r2)

#[과제] --- 완료
#1. scaler 하기 전
# loss :  [1.329974889755249, 0.9134203791618347]
# r2스코어 :  -0.00013399220490351027

#2. MinMaxScaler()
# loss :  [0.3103863000869751, 0.37117621302604675]
# r2스코어 :  0.7665911561100265

#3. StandardScaler()
# loss :  [0.31431177258491516, 0.3840213418006897]
# r2스코어 :  0.7636391857129654

