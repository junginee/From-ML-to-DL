from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sqlalchemy import false
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.keras.utils import to_categorical # https://wikidocs.net/22647 케라스 원핫인코딩
from sklearn.preprocessing import OneHotEncoder  # https://psystat.tistory.com/136 싸이킷런 원핫인코딩
from sklearn.datasets import load_digits
import tensorflow as tf

###########################폴더 생성시 현재 파일명으로 자동생성###########################################
import inspect, os
a = inspect.getfile(inspect.currentframe()) #현재 파일이 위치한 경로 + 현재 파일 명
print(a)
print(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))) #현재 파일이 위치한 경로
print(a.split("\\")[-1]) #현재 파일 명
current_name = a.split("\\")[-1]
##########################밑에 filepath경로에 추가로  + current_name + '/' 삽입해야 돌아감#######################

#1. 데이터

datasets = load_digits()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (1797, 64) (1797,)
print(np.unique(y)) # [0 1 2 3 4 5 6 7 8 9]
print(x,y)

y = to_categorical(y)
print(np.unique(y, return_counts=True)) # y의 라벨값 :  (array([0., 1.], dtype=float32), array([16173,  1797], dtype=int64))

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.7,
                                                    random_state=66
                                                    )

# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델

# model = load_model("./_save/keras22_hamsu07_digit.h5")

# model = Sequential()
# model.add(Dense(30, input_dim=64, activation='linear')) #sigmoid : 이진분류일때 아웃풋에 activation = 'sigmoid' 라고 넣어줘서 아웃풋 값 범위를 0에서 1로 제한해줌
# model.add(Dense(20, activation='sigmoid'))               # 출력이 0 or 1으로 나와야되기 때문, 그리고 최종으로 나온 값에 반올림을 해주면 0 or 1 완성
# model.add(Dense(20, activation='relu'))               # relu : 히든에서만 쓸수있음, 요즘에 성능 젤좋음
# model.add(Dense(20, activation='linear'))               
# model.add(Dense(10, activation='softmax'))             # softmax : 다중분류일때 아웃풋에 활성화함수로 넣어줌, 아웃풋에서 소프트맥스 활성화 함수를 씌워 주면 그 합은 무조건 1로 변함
#                                                                  # ex 70, 20, 10 -> 0.7, 0.2, 0.1

input1 = Input(shape=(64,))
dense1 = Dense(30, activation='linear')(input1)
dense2 = Dense(20, activation='sigmoid')(dense1)
dense3 = Dense(20, activation='relu')(dense2)
dense4 = Dense(20, activation='linear')(dense3)
output1 = Dense(10, activation='softmax')(dense4)
model = Model(inputs=input1, outputs=output1)  

#3. 컴파일 훈련

model.compile(loss='categorical_crossentropy', optimizer='adam', # 다중 분류에서는 로스함수를 'categorical_crossentropy' 로 써준다 (99퍼센트로)
              metrics=['accuracy'])

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M") # 0707_1723
print(date)


filepath = './_ModelCheckPoint/' + current_name + '/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='auto', verbose=1, 
                              restore_best_weights=True)        

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, 
                      filepath= "".join([filepath, date, '_', filename])
                      )

hist = model.fit(x_train, y_train, epochs=1000, batch_size=100,
                 validation_split=0.2,
                 callbacks=[earlyStopping, mcp],
                 verbose=1)

#4. 평가, 예측

results= model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('accuracy : ', results[1])


y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis= 1)
y_predict = to_categorical(y_predict)

acc= accuracy_score(y_test, y_predict)
print('loss : ', results[0])
print('acc스코어 : ', acc) 

#drop out 이전



#drop out 이후