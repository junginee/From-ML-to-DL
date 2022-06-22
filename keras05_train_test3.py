#사이킷런

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10]) 


#[검색]train과 test를 섞어서 7:3으로 찾을 수 있는 방법 찾아라

from sklearn.model_selection import train_test_split
#사이킷런 모델에서 제공하는 train_test_split을 사용한다.

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, train_size=0.7, shuffle=True, random_state=66)

# test_size=0.3, train_size=0.7으로 설정하고 섞을 것이다. 랜덤난수는 66으로 고정한다.랜덤값을 설정하지 않으면 추출되는 데이터는 계속 바뀐다.
# 만약, shuffle을 False로 준다면 앞에서부터 순차적으로 슬라이싱 된 데이터 값이 추출된다.
# test_size / train_size 둘 줄 하나만 기재해도 ok (두개 다 쓸 필요X)

print(x_train) #[2 7 6 3 4 8 5]
print(x_test) #[ 1  9 10]
print(y_train) #[2 7 6 3 4 8 5]
print(x_test) #[ 1  9 10]

#2. 모델구성 
model = Sequential()
model.add(Dense(5,input_dim=1)) 
model.add(Dense(5)) 
model.add(Dense(5))
model.add(Dense(7))
model.add(Dense(9))  
model.add(Dense(1))  

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit( x_train, y_train, epochs=300, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test,y_test,)
print('loss :',loss)
result = model.predict([11]) 
print('11의 예측값:', result)







