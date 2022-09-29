import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, LSTM

#1. 데이터
a = np.array(range(1,101))
size = 5 #x는 4개


def split_x(dataset,size):
    aaa = []
    for i in range(len(dataset)-size + 1):
        subset = dataset[i : (i + size)]
        aaa.append(subset)
    return np.array(aaa)

bbb = split_x(a, size)

x = bbb[:, :-1] 
y = bbb[:, -1]

print(x,y)
print(x.shape,y.shape) #(96, 4) (96,)

x = x.reshape(96,4,1)
print(x.shape) #(96, 4, 1)

#---------------------------------------------
x_predict = np.array(range(96,106)) 
size = 4

def split_x_predict(dataset,size):
    ccc = []
    for i in range(len(dataset)-size + 1):
        subset = dataset[i : (i + size)]
        ccc.append(subset)
    return np.array(ccc)

ddd = split_x_predict(x_predict, size)
print(ddd)
print(ddd.shape) #(7, 4)

x_predict = ddd[:, :] 



#2. 모델구성
model= Sequential()                                                                                         
model.add(LSTM(units = 64,input_shape=(4,1))) 
model.add(Dense(16, activation = 'relu')) #rnn은 cnn과 달리 2차원으로 모델 구성되기 때문에 flatten 사용x
model.add(Dense(32, activation = 'relu')) 
model.add(Dense(32)) 
model.add(Dropout(0.15))
model.add(Dense(50)) 
model.add(Dropout(0.2))
model.add(Dense(1))

model.summary()

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=700)

# 4. 평가, 예측
loss = model.evaluate(x,y)
x_predict = x_predict.reshape(7,4,1) 
print(x_predict)

result = model.predict (x_predict)

print('loss :',loss)
print('range(96,106) : ', result)

# [[ 99.05081]
#  [ 99.53784]
#  [ 99.9417 ]
#  [100.32264]
#  [100.68415]
#  [100.98461]
#  [101.24826]]

