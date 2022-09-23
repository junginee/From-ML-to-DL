import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#1. 데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
             [1, 1, 1, 1, 2, 1.3, 1.4, 1.5, 1.6, 1.4],
             [9,8,7,6,5,4,3,2,1,0]]
            )
y = np.array([11,12,13,14,15,16,17,18,19,20])   

print(x.shape, y.shape)  #(3,10) #(10,)
x = x.transpose()
print(x.shape)#(10,3)

#2. 모델구성
# model = Sequential()
# #model.add(Dense(10, input_dim=3))
# model.add(Dense(10, input_shape=(3,)))
# model.add(Dense(5))
# model.add(Dense(3))
# model.add(Dense(1))

model = Sequential()
model.add(Dense(10, input_shape=(3,)))
model.add(Dense(5))
model.add(Dense(3))
model.add(Dense(1))
'''
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 dense (Dense)               (None, 10)                40

 dense_1 (Dense)             (None, 5)                 55

 dense_2 (Dense)             (None, 3)                 18

 dense_3 (Dense)             (None, 1)                 4

=================================================================
Total params: 117
Trainable params: 117
Non-trainable params: 0 
_________________________________________________________________

'''
