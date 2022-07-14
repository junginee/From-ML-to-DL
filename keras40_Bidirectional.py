import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Dropout 
from tensorflow.keras.layers import Bidirectional

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
model.add(SimpleRNN(10,return_sequences = True,input_shape=(3,1))) #행 무시(행은 모델을 구성하는데 관여 x)
model.add(Bidirectional(SimpleRNN(5)))
model.add(Dense(3, activation = 'relu')) 
model.add(Dense(1))

model.summary()

# Model: "sequential"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  simple_rnn (SimpleRNN)      (None, 3, 10)             120

#  bidirectional (Bidirectiona  (None, 10)               160
#  l)

#  dense (Dense)               (None, 3)                 33

#  dense_1 (Dense)             (None, 1)                 4

# =================================================================
# Total params: 317
# Trainable params: 317
# Non-trainable params: 0
# _________________________________________________________________

# 5*(5+10+1) * 2 = 160