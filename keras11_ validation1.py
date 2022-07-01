#검증?


from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np

#1. 데이터
x_train = np.array(range(1,11))
y_train = np.array(range(1,11))
x_test = np.array([11,12,13]) #evaluate에서 사용할 변수
y_test = np.array([11,12,13])
x_val = np.array([14,15,16])
y_val  = np.array([14,15,16])

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

#val_loss : loss값보다 더 좋지(값이 떨어지지) 않아야한다. 왜? validation loss를 이용하여 overfitting을 방지
#우리의 목적은 학습을 통해 머신 러닝 모델의 underfitting된 부분을 제거해나가면서 overfitting이 발생하기 직전에 학습을 멈추는 것이다. 이를 위해 머신 러닝에서는 validation set을 이용한다.
#https://untitledtblog.tistory.com/158
