

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import numpy as np

#1. 데이터
x = np.array(range(1,17))
y = np.array(range(1,17))

#[실습] train_test_split으로만 나눠라! 
# # 10 : 3 : 3으로 나눠라 

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.625,  shuffle=True, random_state=66) # 16/10 = 0.625 >> train 0.625로 줌으로써 10개의 값 추출
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, shuffle=True, random_state=70) #test 값(6개) 중 3개는 test, 3개는 validation으로 준다.

print(x_train) #10개
print(x_test) #3개
print(x_val) #3개

# x_train = np.array(range(1,11))
# y_train = np.array(range(1,11))
# x_test = np.array([11,12,13]) #evaluate에서 사용할 변수
# y_test = np.array([11,12,13])
# x_val = np.array([14,15,16])
# y_val  = np.array([14,15,16])

# #2. 모델구성
# model = Sequential()
# model.add(Dense(5,input_dim = 1))
# model.add(Dense(3))


# #3. 컴파일, 훈련
# model.compile(loss ='mse', optimizer='adam' )
# model.fit(x_train, y_train, epochs = 100, batch_size = 1, validation_data=(x_val, y_val))
         
# #4. 평가, 예측
# loss = model.evaluate(x_test,y_test)
# print('loss :', loss)

# result = model.predict([17])
# print("17의 예측값 : ", result)


