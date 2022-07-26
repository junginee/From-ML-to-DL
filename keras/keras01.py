#케라스모델 
#1.데이터
import numpy as np
x = np.array([1,2,3])
y = np.array([1,2,3])            
                       
#2.모델구성
from tensorflow.keras.models import Sequential #내가 사용할 모델은 시퀀셔 모델이다.
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(4, input_dim=1)) #첫번째 밀집층을 더한다. (=레이어층 하나를 추가시킨다.) 입력하는 차원은 1차원이다. (=인풋레이어에 들어가는 데이터의 형태) #노드 4 :아웃풋은 4
model.add(Dense(5)) #두번째 레이어층의 인풋값은 4, 아웃풋값은 5 (시퀀셔 모델을 이용하기 때문에 자동으로 인풋값은 4가 된다.)
model.add(Dense(3)) #세번째 레이어층의 인풋값은 5, 아웃풋값은 3
model.add(Dense(3)) #네번째 레이어층의 인풋값은 3, 아웃풋값은 2
model.add(Dense(1)) #다섯번째 레이어층의 인풋값은 2, 아웃풋값은 1

#3. 컴파일, 훈련                            #로스에 대한 최적화는 adam을 사용할 것이다. 
model.compile(loss='mse',optimizer='adam') #mse=평균제곱오차(로스는 mse를 사용하겠다. 오차 계산 시 음수가 있으면 상쇄되기 때문에 이것을 방지하기 위해 평균제곱오차를 사용하겠다.)
model.fit(x,y,epochs=600) #1000번 훈련시키겠다. 이때 훈련을 여러번, 많이 시킨다고 최적의 값이 나오는 것은 아니다. #이곳에 최종 갱신된 가중치가 보관

#4. 평가, 예측
loss = model.evaluate(x, y) #x,y의 평가값은 로스입니다.
print('loss :', loss)  #로스값을 출력해주세요.    

result = model.predict([4]) #예측값으로 4를 주며, 4는 result에 저장한다.
print('4의 예측값 : ', result)
