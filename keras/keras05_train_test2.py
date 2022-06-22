#슬라이싱

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10]) 

#[과제]넘파이 리스트의 슬라이싱 7:3으로 잘라라
x_train = x[0:7]
x_test = x[7:10]
y_train = x[:7]
y_test = x[7:10]

print(x_train)
print(x_test)
print(y_train)
print(x_test)

# Q. 슬라이싱을 7:3으로 할 경우 어떤 문제가 있을까?
# A. 우리가 찾아야하는 데이터는 1부터 10까지의 범위 
# BUT train으로 1부터 7까지의 범위를 줌 이럴경우 데이터 한쪽으로 편향될 수 있음. 
# 이와 같이 데이터 편향을 막으며 test는 30%를 추출하려면 어떻게 해야할까?
# 30%의 데이터를 뒤에서부터 30%만 빼는 것이 아닌 중간중간 랜덤으로 추출한다.
# how? 랜덤함수로 데이터를 섞어준 후, test set을 중간중간 랜덤 추출 

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

#데이터를 평가하려면 훈련되지 않은 데이터로 평가해야한다.





