#검증?


from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array(range(1,17))
y = np.array(range(1,17))

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, train_size=0.4, shuffle=True, random_state=66)
x_test, x_val, y_test, y_val = train_test_split(x, y, test_size=0.5, shuffle=True, random_state=70)

# x_train = np.array(range(1,11))
# y_train = np.array(range(1,11))
# x_test = np.array([11,12,13]) #evaluate에서 사용할 변수
# y_test = np.array([11,12,13])
# x_val = np.array([14,15,16])
# y_val  = np.array([14,15,16])

#2. 모델구성
model = Sequential()
model.add(Dense(5,input_dim = 1))
model.add(Dense(3))


#3. 컴파일, 훈련
model.compile(loss ='mse', optimizer='adam' )
model.fit(x_train, y_train, epochs = 100, batch_size = 1, validation_data=(x_val, y_val))
         
#4. 평가, 예측
loss = model.evaluate(x_test,y_test)
print('loss :', loss)

result = model.predict([17])
print("17의 예측값 : ", result)


