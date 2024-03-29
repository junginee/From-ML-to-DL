import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, SimpleRNN, Dropout

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
model= Sequential()                                                                                         #input_dim
#model.add(SimpleRNN(units = 10,input_shape=(3,1))) #행 무시(행은 모델을 구성하는데 관여 x) #[batch, timesteps, feature]
model.add(SimpleRNN(units = 10,input_length=3, input_dim = 1)) #행 무시(행은 모델을 구성하는데 관여 x) #[batch, timesteps, feature]
                      
model.add(Dense(5, activation = 'relu')) #rnn은 cnn과 달리 2차원으로 모델 구성되기 때문에 flatten 사용x
model.add(Dense(1))

model.summary()

# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# simple_rnn (SimpleRNN)       (None, 10)                120
# _________________________________________________________________
# dense (Dense)                (None, 5)                 55
# _________________________________________________________________
# dense_1 (Dense)              (None, 1)                 6
# =================================================================
# Total params: 181
# Trainable params: 181
# Non-trainable params: 0
# _________________________________________________________________
