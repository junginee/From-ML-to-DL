import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Dropout,Flatten, Conv1D #3차원

#1. 데이터 
datasets = np.array([1,2,3,4,5,6,7,8,9,10])

x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8],[7,8,9]])
#(N,3)
y = np.array([4,5,6,7,8,9,10])

#x의 shape = (행, 열, 몇개씩 자르는지) = 3차원

print(x.shape, y.shape) #(7, 3) (7,)

x = x.reshape(7,3,1)
print(x.shape) #(7, 3, 1)

#2. 모델구성
model= Sequential()
#model.add(LSTM(10,return_sequences = False,input_shape=(3,1))) #행 무시(행은 모델을 구성하는데 관여 x)
model.add(Conv1D(10,2,input_shape=(3,1))) #행 무시(행은 모델을 구성하는데 관여 x)
model.add(Flatten())
model.add(Dense(3, activation = 'relu')) 
model.add(Dense(1))

model.summary()
