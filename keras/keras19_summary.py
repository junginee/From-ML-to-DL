from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array([1,2,3])
y = np.array([1,2,3])

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim = 1))
model.add(Dense(3, activation = 'relu'))
model.add(Dense(4, activation = 'sigmoid'))
model.add(Dense(2, activation = 'relu'))
model.add(Dense(1))

model.summary()

'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 5)                 10                 (node 1 + bias node 1) * 5 = 10
_________________________________________________________________
dense_1 (Dense)              (None, 3)                 18                 (node 5 + bias node 1) * 3 = 18
_________________________________________________________________
dense_2 (Dense)              (None, 4)                 16                 (node 3 + bias node 1) * 4 = 16
_________________________________________________________________
dense_3 (Dense)              (None, 2)                 10                 (node 4 + bias node 1) * 2 = 10
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 3                  (node 2 + bias node 1) * 1 = 3
=================================================================
  

'''
