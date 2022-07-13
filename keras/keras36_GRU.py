#DNN 2차원 RNN 3차원 CNN 4차원
#RNN은 왜 3차원? (N,3) > 몇개씩 자르는지 갯수 추가 > 예 : (N,3,몇개씩 자르는지) = 3차원

import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, SimpleRNN, Dropout, LSTM, GRU


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
model.add(GRU(units = 64,input_shape=(3,1))) 
model.add(Dense(16, activation = 'relu')) #rnn은 cnn과 달리 2차원으로 모델 구성되기 때문에 flatten 사용x
model.add(Dense(16, activation = 'relu')) 
model.add(Dense(32)) 
model.add(Dropout(0.15))
model.add(Dense(50)) 
model.add(Dropout(0.2))
model.add(Dense(1))

model.summary()

# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# gru (GRU)                    (None, 64)                12672
# _________________________________________________________________
# dense (Dense)                (None, 16)                1040
# _________________________________________________________________
# dense_1 (Dense)              (None, 16)                272
# _________________________________________________________________
# dense_2 (Dense)              (None, 32)                544
# _________________________________________________________________
# dropout (Dropout)            (None, 32)                0
# _________________________________________________________________
# dense_3 (Dense)              (None, 50)                1650
# _________________________________________________________________
# dropout_1 (Dropout)          (None, 50)                0
# _________________________________________________________________
# dense_4 (Dense)              (None, 1)                 51
# =================================================================
# Total params: 16,229
# Trainable params: 16,229
# Non-trainable params: 0
# _________________________________________________________________


# units * (feature + bias + units) = parms

# [simple] units : 10 -> 10 * (1 + 1 + 10) = 120
# [LSTM] units : 10 -> 4 * 10 * (1 + 1 + 10) = 480
# 결론 : LSTM = simpleRNN * 4
# [GRU] units : 10 -> 3 * 10 * (1 + 1 + 10) = 360

# LSTM 결론 : simpleRNN * 4
# 숫자4의 의미는 cell state, input gate, forget gate, output gate

# GRU 결론 : simpleRNN * 3
# 숫자3의 의미는 hidden state, update gate, reset gate


# #3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')
# model.fit(x,y, epochs=850)

# #4. 평가, 예측
# loss = model.evaluate(x,y)
# y_pred = np.array([8,9,10]).reshape(1,3,1) #[[[8], [9], [10]]]


# result = model.predict (y_pred)

# print('loss :', loss)
# print('[8,9,10] : ', result)


# loss : 0.3139902949333191
# [8,9,10] :  [[9.968759]]