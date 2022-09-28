import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, SimpleRNN, Dropout, LSTM


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
model.add(LSTM(units = 64,input_shape=(3,1))) 
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
# lstm (LSTM)                  (None, 10)                480
# _________________________________________________________________
# dense (Dense)                (None, 5)                 55
# _________________________________________________________________
# dense_1 (Dense)              (None, 1)                 6
# =================================================================
# Total params: 541
# Trainable params: 541
# Non-trainable params: 0
# _________________________________________________________________


# units * (feature + bias + units) = parms

# [simple] units : 10 -> 10 * (1 + 1 + 10) = 120
# [LSTM] units : 10 -> 4 * 10 * (1 + 1 + 10) = 480
# 결론 : LSTM = simpleRNN * 4
# 숫자4의 의미는 cell state, input gate, output gate, forget gate


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=850)

#4. 평가, 예측
loss = model.evaluate(x,y)
y_pred = np.array([8,9,10]).reshape(1,3,1) #[[[8], [9], [10]]]


result = model.predict (y_pred)

print('loss :', loss)
print('[8,9,10] : ', result)

# loss : 0.01725155860185623
# [8,9,10] :  [[10.982748]]
