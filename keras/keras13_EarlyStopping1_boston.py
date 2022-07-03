import numpy as np 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score

#1.데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

#2.모델구성
model = Sequential()
model.add(Dense(5, input_dim=13))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(5))
model.add(Dense(7))
model.add(Dense(1))
import time    #시간
 
#3.컴파일,훈련
model.compile(loss='mse', optimizer='adam')

#early stopping
from tensorflow.python.keras.callbacks import EarlyStopping
######## earlystopping를 정의 (*사용한것 아님) ##################
earlyStopping = EarlyStopping(monitor='loss', patience=10, mode= 'min', #min 값은 디폴트 값이 auto(loss계열은 최소값, acc계열은 최대값으로 자동 지정)
                              verbose=1,restore_best_weights=True)# 최소값을 10번 갱신 못하면 끊겠다. restore:디폴트값은 false(스탑한 지점의 weight) true(최소값이 있는 지점의 w)
###############################################################

start_time = time.time()
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size=0.2,
                                                    shuffle=True,
                                                    random_state=66)

hist = model.fit(x_train, y_train, epochs=1000, batch_size=1, verbose=1,
                  validation_split = 0.2,
                  callbacks=[earlyStopping]) 

end_time = time.time() - start_time

#4.평가,예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
y_predict = model.predict(x_test) 



################### history ############################
print('==============================')
print(hist)
print('==============================')
print(hist.history)    #딕셔너리 키:loss , 벨류2개(loss,val loss): 리스트 형태로 저장되있음, 11번 훈련 저장함
print('==============================')
print(hist.history['loss'])# loss 만 출력 (키 벨류 ' ' 해야됨.)
print('==============================')
print(hist.history['val_loss'])#val_loss 만 출력


print('걸린시간 : ', end_time)

#R2결정계수(성능평가지표)
r2 = r2_score(y_test, y_predict) 
print('r2스코어 : ' , r2) 

##################### 한글 깨짐 해결 ###################
import matplotlib
matplotlib.rcParams['font.family'] ='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] =False
#######################################################

import matplotlib.pyplot as plt
plt.figure(figsize=(9,6)) #그래프 표 사이즈
plt.plot(hist.history['loss'], marker = '.' ,c = 'red', label = 'loss') # maker: 점으로 표시하겟다  ,c:색깔 ,label : 이름
plt.plot(hist.history['val_loss'], marker = '.' ,c = 'blue', label = 'val_loss')
plt.grid() # 모눈종이에 하겠다
plt.title('keras13_EarlyStopping1_boston')#제목
plt.ylabel('loss')#y축 이름
plt.xlabel('epochs')#x축 이름
plt.legend(loc='upper right') # upper right: 위쪽 상단에 표시하겠다.(라벨 이름들)
plt.show()# 보여줘

#loss와 val_loss의 간격이 좁은게 loss 자체가 낮은것보다 오히려 좋다 
#훈련 중 loss최솟값을 계속 체크, n번의 횟수이상 최솟값이 안나오면 강제 중지 : early stopping
# r2스코어 :  0.7591639861005413
