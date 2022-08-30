import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

#1.데이터
x_data = [[0,0],[0,1],[1,0],[1,1]]
y_data = [0,1,1,0] 

#2.모델
model = Sequential()
model.add(Dense(10, input_dim = 2))
model.add(Dense(10))
model.add(Dense(12))
model.add(Dense(5, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

#3.훈련
model.compile(loss = 'binary_crossentropy', optimizer = 'adam',metrics = ['acc'])
model.fit(x_data, y_data, batch_size = 1, epochs=100)

#4.평가, 예측
y_predict = model.predict(x_data)
print(x_data, "의 예측결과 : ", y_predict)

results = model.evaluate(x_data, y_data)
print("model.evaluate :", results[1])
