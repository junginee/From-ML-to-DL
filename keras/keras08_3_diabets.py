# [실습]
# R2 0.62 이상 
#y는 정제된 x의 데이터 값으로 나온 결과치기 때문에 정제를 할 필요는 없다.


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

datasets =load_diabetes()
x = datasets.data
y = datasets.target
x_train,x_test,y_train, y_test = train_test_split(x,y, train_size=0.75, shuffle=True, random_state=72)

# print(x)
# print(y) 
# print(x.shape, y.shape) #(442, 10) (442,)
# print(datasets.feature_names)
# print(datasets.DESCR)


#2.모델구성
model = Sequential()
model.add(Dense(6,input_dim=10))
model.add(Dense(15))
model.add(Dense(15))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(1))

#3.컴파일, 훈련
model.compile(loss='mae', optimizer = 'adam')
model.fit(x_train, y_train , epochs=500, batch_size=10)

#4.평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ',loss)

y_predict = model.predict(x_test) 
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict) #R 제곱 = 예측값 (y_predict) / 실제값 (y_test) 
print('r2스코어 : ', r2)

#결과
#loss :  39.40278625488281
#r2스코어 :  0.6281169407558016



