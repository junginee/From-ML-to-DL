import numpy as np 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt

#1.데이터     
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

#2.모델구성
model = Sequential()
model.add(Dense(50, input_dim=8)) 
model.add(Dense(50,activation ='relu'))
model.add(Dense(50,activation ='relu'))
model.add(Dense(50,activation ='relu'))
model.add(Dense(50,activation ='relu'))
model.add(Dense(1))

#3.컴파일,훈련
model.compile(loss='mse', optimizer='adam')

from tensorflow.python.keras.callbacks import EarlyStopping

earlyStopping = EarlyStopping(monitor= 'val_loss',patience=30,mode= 'min', restore_best_weights=True,verbose=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    train_size=0.7,
                                                    shuffle=True,
                                                    random_state=100)

hist = model.fit(x_train, y_train, epochs=2000, batch_size=50,
         validation_split= 0.2,callbacks= earlyStopping)

#4.평가,예측
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict) 
print('r2스코어 : ' , r2) 


##################### 한글 깨짐 해결 ###################
import matplotlib
matplotlib.rcParams['font.family'] ='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] =False
#######################################################

plt.figure(figsize=(9,6)) #그래프 표 사이즈
plt.plot(hist.history['loss'], marker = '.' ,c = 'red', label = 'loss') # maker: 점으로 표시하겟다  ,c:색깔 ,label : 이름
plt.plot(hist.history['val_loss'], marker = '.' ,c = 'blue', label = 'val_loss')
plt.grid() # 모눈종이에 하겠다
plt.title('keras13_EarlyStopping2_california')#제목
plt.ylabel('loss')#y축 이름
plt.xlabel('epochs')#x축 이름
plt.legend(loc='upper right') # upper right: 위쪽 상단에 표시하겠다.(라벨 이름들)
plt.show()# 보여줘

#patience100번은 너무 많다..epochs 1000번에 earlystopping이 안됨

# loss :  0.47151753306388855   
# r2스코어 :  0.6504332447145813
