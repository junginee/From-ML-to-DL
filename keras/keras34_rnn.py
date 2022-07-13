#DNN 2차원 RNN 3차원 CNN 4차원
#RNN은 왜 3차원? (N,3) > 몇개씩 자르는지 갯수 추가 > 예 : (N,3,몇개씩 자르는지) = 3차원

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
model= Sequential()
model.add(SimpleRNN(64,input_shape=(3,1))) #행 무시(행은 모델을 구성하는데 관여 x)
model.add(Dense(16, activation = 'relu')) #rnn은 cnn과 달리 2차원으로 모델 구성되기 때문에 flatten 사용x
model.add(Dense(16, activation = 'relu')) 
model.add(Dense(32)) 
model.add(Dropout(0.15))
model.add(Dense(35)) 
model.add(Dropout(0.2))
model.add(Dense(1))



#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=850)

#4. 평가, 예측
loss = model.evaluate(x,y)
y_pred = np.array([8,9,10]).reshape(1,3,1) #[[[8], [9], [10]]]


result = model.predict (y_pred)

print('loss :', loss)
print('[8,9,10] : ', result)


# loss : 0.006160564720630646
# [8,9,10] :  [[10.962086]]