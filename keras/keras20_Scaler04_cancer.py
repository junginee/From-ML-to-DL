#[과제] --- 완료
#1. scaler 하기 전
# loss :  [0.2606964707374573, 0.9122806787490845]
# acc스코어 :  0.9122807017543859

#2. MinMaxScaler()
# 걸린시간 :  1657091060.419887
# loss :  [0.15346671640872955, 0.9590643048286438]
# acc스코어 :  0.9590643274853801

#3. StandardScaler()
# 걸린시간 :  1657091174.0220242
# loss :  [0.11797408759593964, 0.9590643048286438]
# acc스코어 :  0.9590643274853801



import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

import matplotlib.pyplot as plt
import time

#1. 데이터
datasets = load_breast_cancer()
# print(datasets) (569,30)
# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets.data   #['data']
y = datasets.target #['target']
print(x.shape, y.shape) # (569,30), (569,)


x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.7,shuffle=True, random_state=72)
###############스캘러 방법 2가지###############################
scaler = StandardScaler()
#Scaler = MinMaxScaler()
scaler .fit(x_train)
x_train = scaler .transform(x_train) 
x_test = scaler .transform(x_test)
print(np.min(x_train)) #0.0 
print(np.max(x_train)) #1.0                                                  random_state=66
                                                    


#2. 모델구성
model = Sequential()
model.add(Dense(30, input_dim=30, activation='linear')) #sigmoid : 이진분류일때 아웃풋에 activation = 'sigmoid' 라고 넣어줘서 아웃풋 값 범위를 0에서 1로 제한해줌
model.add(Dense(20, activation='sigmoid'))               # 출력이 0 or 1으로 나와야되기 때문, 그리고 최종으로 나온 값에 반올림을 해주면 0 or 1 완성
model.add(Dense(20, activation='relu'))               # relu : 히든에서만 쓸수있음, 요즘에 성능 젤좋음
model.add(Dense(20, activation='linear'))               
model.add(Dense(1, activation='sigmoid'))   
                                                                        
#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])   # 이진분류에 한해 로스함수는 무조건 99퍼센트로 'binary_crossentropy'
                                      # 컴파일에있는 metrics는 평가지표라고도 읽힘

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=200, mode='auto', verbose=1, 
                              restore_best_weights=True)        

                  #restore_best_weights false 로 하면 중단한 지점의 웨이트값을 가져옴 true로하면 끊기기 전이라도 최적의 웨이트값을 가져옴


start_time = time.time()
hist = model.fit(x_train, y_train, epochs=3000, batch_size=100,
                 validation_split=0.3,
                 callbacks=[earlyStopping],
                 verbose=1)

end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

#### 과제 1 accuracy_score 완성 y 테스트는 반올림이 됫는데 y 프리딕트는 반올림이 안됫음 ######
y_predict = y_predict.round(0)
print(y_predict)

print("걸린시간 : ", end_time)

acc= accuracy_score(y_test, y_predict)
print('loss : ' , loss)
print('acc스코어 : ', acc) 

