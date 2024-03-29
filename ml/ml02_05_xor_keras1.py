import numpy as np
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

# 1. 데이터
x_data = [[0,0],[0,1],[1,0],[1,1]]
y_data = [0,1,1,0] # xor: 같으면 0 다르면 1

# 2. 모델
# model = LinearSVC()
# model = Perceptron()
# model = SVC()
model = Sequential()
model.add(Dense(1, input_dim=2, activation='sigmoid'))

# 3. 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_data, y_data, batch_size=1, epochs=100)

# 4. 평가, 예측
y_pred = model.predict(x_data)
print(x_data, '의 예측결과: ', y_pred)

result = model.evaluate(x_data, y_data)
print('model.score: ', result[1])


# [[0, 0], [0, 1], [1, 0], [1, 1]] 의 예측결과:  [[0.47643304]
#  [0.57909226]
#  [0.63553977]
#  [0.72500855]]
# model.score:  0.75
