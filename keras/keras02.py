#[실습] [6]을 예측한다.



#1. 데이터
import numpy as np
x = np.array([1,2,3,5,4])
y = np.array([1,2,3,4,5])

#2.모델구성
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(4, input_dim=1)) 
model.add(Dense(10))
model.add(Dense(1)) 

#3. 컴파일, 훈련                            
model.compile(loss='mae',optimizer='adam') 
model.fit(x,y,epochs=300) 

#4. 평가, 예측
loss = model.evaluate(x, y) 
print('loss :', loss) 

result = model.predict([6]) 
print('6의 예측값 : ', result)


#나는 배운 내용을 토대로 로스에 회귀모델 mse로 설정하였다. 최대 예측치는 5.8정도로 6과 가까운 예측치가 나오지 않았다
#mse를 mae(평균제곱오차)로 설정했다.       
         





        
