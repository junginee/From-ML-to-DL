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
model.add(GRU(units = 64,input_shape=(3,1))) 
model.add(Dense(32, activation = 'relu')) 
model.add(Dense(32, activation = 'relu')) 
model.add(Dense(40)) 
model.add(Dropout(0.15))
model.add(Dense(50)) 
model.add(Dropout(0.1))
model.add(Dense(1))


# #3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=850)

#4. 평가, 예측
loss = model.evaluate(x,y)
y_predict = np.array([50,60,70]).reshape(1,3,1) #[[[8], [9], [10]]]

result = model.predict (y_predict)

print('loss :', loss)
print('[8,9,10] : ', result)

# loss : 2.1010031700134277
# [8,9,10] :  [[70.629974]]
