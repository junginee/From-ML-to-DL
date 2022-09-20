#나쁜모델 판별하기
#레이어 수가 많을수록, 한 레이어 당 노드 수가 많을수록 나쁜모델이 구현

#1.R2를 음수가 아닌 0.5 이하로 만들어라.
#2.데이터는 건들이지 않는다.
#3.레이어는 인풋, 아웃풋 포함 7개 이상
#4.batch_size=1
#5. 히든레이어의 노드는 10개 이상 100개 이하
#6. train 70%
#7. epoch 100번 이상

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
y = np.array([1,2,4,3,5,7,9,3,8,12,13,8,14,15,9,6,17,23,21,20])
x_train, x_test, y_train, y_test = train_test_split(x,y,
          train_size=0.7, shuffle=False, random_state=50)

#2. 모델구성
model=Sequential()
model.add(Dense(2, input_dim=1))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(100))
model.add(Dense(1))

           
#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ',loss)

y_predict = model.predict(x)
from sklearn.metrics import r2_score
r2 = r2_score(y, y_predict) #R 제곱 = 예측값 (y_predict) / 실제값 (y)
print('r2스코어 : ', r2)

#[결과]
#loss :  28.602506637573242
#r2스코어 :  0.4791184589890901
