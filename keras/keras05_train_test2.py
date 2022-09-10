#슬라이싱

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10]) 

#넘파이 리스트의 슬라이싱 
x_train = x[0:7]
x_test = x[7:10]
y_train = x[:7]
y_test = x[7:10]

print(x_train)
print(x_test)
print(y_train)
print(x_test)

#2. 모델구성 
model = Sequential()
model.add(Dense(5,input_dim=3)) 
model.add(Dense(5)) 
model.add(Dense(5))
model.add(Dense(7))
model.add(Dense(9))  
model.add(Dense(1))  

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=300, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x,y)
print('loss :',loss)
result = model.predict([[10, 1.4, 0]]) #(1,3)
print('[10, 1.4]의 예측값:', result)
