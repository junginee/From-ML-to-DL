

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time

datasets = fetch_california_housing() 
x = datasets.data 
y = datasets.target 
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, shuffle=True, random_state=50)


scaler = MinMaxScaler()
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

model.save_weights("./_save/keras23_8_save_California.h5")

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ',loss)

print("걸린시간 : ", end_time)
y_predict = model.predict(x_test) 

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict) #R 제곱 = 예측값 (y_predict) / 실제값 (y_test) 
print('r2스코어 : ', r2)
