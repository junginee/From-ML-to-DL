#verbose ; 훈련과정 생략 > 장점 : 빠르게 처리할 수 있다. (시간 지체 현상이 적게 나타난다.)
#verbose를 0으로 하고(훈련과정이 출력되는 것을 생략하고), 훈련시간이 얼마나 걸렸는지 확인해보자!  >> (1)import time (2)start_time = time.time() (3)print(start_time)
                                                                                             #(4)end_time = time.time() - start_time

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston 

# 1. 데이터
datasets = load_boston() 
x = datasets.data 
y = datasets.target 
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.70, shuffle=True, random_state=88)

# 2. 모델구성

model = Sequential()
model.add(Dense(5, input_dim=13))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(7))
model.add(Dense(1))


# 3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam')

import time 
start_time = time.time()
print(start_time)

model.fit(x_train, y_train, epochs=50, batch_size=7, verbose=0)
end_time = time.time() - start_time

print("걸린시간 : ", end_time)

"""
vervose 0 걸린시간 : 출력없다. (진행상황을 볼 수 없다.)
vervose 1 걸린시간 : 잔소리많다.
vervose 2 걸린시간 : 프로그래스바 없다.
vervose 3,4,5... 걸린시간 : epoch만 나온다. 
"""  
