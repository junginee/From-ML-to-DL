import numpy as np
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, SimpleRNN, Dropout, LSTM, GRU
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
             [5,6,7],[6,7,8],[7,8,9],[8,9,10],
             [9,10,11],[10,11,12],
             [20,30,40],[30,40,50],[40,50,60]]) 
y = np.array([4,5,6,7,8,9,10,1,12,13,50,60,70])


print(x.shape, y.shape) #(13, 3) (13,)

x = x.reshape(13,3,1)
print(x.shape) #(13, 3, 1)


#2. 모델구성
model= Sequential()                                                                                         
model.add(LSTM(10, return_sequences=True, input_shape=(3,1))) #(N,3,1) > (N,3,10)
#return_sequences=True 를 설정한다면, rnn의 output이 2차원 > 3차원으로 변형된다. 
#so, LSTM 모델끼리 연결할 수 있다.
model.add(LSTM(5)) 
model.add(Dense(1))

model.summary()

# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# lstm (LSTM)                  (None, 3, 10)             480
# _________________________________________________________________
# lstm_1 (LSTM)                (None, 5)                 320
# _________________________________________________________________
# dense (Dense)                (None, 1)                 6
# =================================================================
# Total params: 806
# Trainable params: 806
# Non-trainable params: 0
# _________________________________________________________________

# # #3. 컴파일, 훈련
# model.compile(loss='mse', optimizer='adam')
# model.fit(x,y, epochs=850)

# #4. 평가, 예측
# loss = model.evaluate(x,y)
# y_predict = np.array([50,60,70]).reshape(1,3,1) #[[[8], [9], [10]]]


# result = model.predict (y_predict)

# print('loss :', loss)
# print('[8,9,10] : ', result)

# # loss : 2.1010031700134277
# # [8,9,10] :  [[70.629974]]
