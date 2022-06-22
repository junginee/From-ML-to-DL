#회귀모델 성능평가지표 R2

#loss값만으로 모델이 좋은지 나쁜지 평가할 수 없기 때문에(상대적이기 때문) r2 성능평가지표를 이용해 이 모델을 평가한다. (정확도를 평가하는 것은 accuraccy 모델=분류모델)
#결정계수 = '회귀 모델의 성과 지표' /1에 가까울 수록 좋은 회귀 모델,0에 가까울 수록 나쁜 모델/음수가 나올경우, 바로 폐기해야 하는 모델
#r2가 1에 가까울수록 loss값은 낮아진다.
#분산 기반으로 예측 성능을 평가한다. 
#실제 값의 분산 대비 예측값의 분산 비율을 지표로 하며, 1에 가까울 수록 예측 정확도가 높다. R 제곱 = 예측값 (y_predict) / 실제값 (y)


from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])
x_train, x_test, y_train, y_test = train_test_split(x,y,
          train_size=0.7, shuffle=True, random_state=66)

#2. 모델구성
model=Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(8))
model.add(Dense(1))
           
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=650, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ',loss)

y_predict = model.predict(x)
from sklearn.metrics import r2_score
r2 = r2_score(y, y_predict)
print('r2스코어 : ', r2)




