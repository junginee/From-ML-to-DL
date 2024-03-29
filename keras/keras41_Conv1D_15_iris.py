from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sqlalchemy import false
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, LSTM, Flatten, Conv1D
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.keras.utils import to_categorical # https://wikidocs.net/22647 케라스 원핫인코딩
from sklearn.preprocessing import OneHotEncoder  # https://psystat.tistory.com/136 싸이킷런 원핫인코딩.
import tensorflow as tf
tf.random.set_seed(66)  # y=wx 할때 w는 랜덤으로 돌아가는데 여기서 랜덤난수를 지정해줄수있음


#1. 데이터
datasets = load_iris()
x = datasets['data']
y = datasets['target']
y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=66
                                                    )
print(x_train.shape, x_test.shape) #(120, 4) (30, 4)

scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(120,2,2)
x_test = x_test.reshape(30, 2,2)
print(x_train.shape, x_test.shape) #(120, 2, 2) (30, 2, 2)
 
#2. 모델

# model = Sequential()
# model.add(Dense(30, input_dim=4, activation='linear')) #sigmoid : 이진분류일때 아웃풋에 activation = 'sigmoid' 라고 넣어줘서 아웃풋 값 범위를 0에서 1로 제한해줌
# model.add(Dense(20, activation='sigmoid'))               # 출력이 0 or 1으로 나와야되기 때문, 그리고 최종으로 나온 값에 반올림을 해주면 0 or 1 완성
# model.add(Dense(20, activation='relu'))               # relu : 히든에서만 쓸수있음, 요즘에 성능 젤좋음
# model.add(Dense(20, activation='linear'))               
# model.add(Dense(3, activation='softmax'))             # softmax : 다중분류일때 아웃풋에 활성화함수로 넣어줌, 아웃풋에서 소프트맥스 활성화 함수를 씌워 주면 그 합은 무조건 1로 변함
                                                                 # ex 70, 20, 10 -> 0.7, 0.2, 0.1
                                                               
input1 = Input(shape=(2,2))
dense1 = Conv1D(10, kernel_size=(2))(input1)
dense2 = Flatten()(dense1)
dense3 = Dense(20, activation='sigmoid')(dense2)
dense4 = Dense(20, activation='relu')(dense3)
dense5 = Dense(20, activation='linear')(dense4)
output1 = Dense(3, activation='softmax')(dense5)
model = Model(inputs=input1, outputs=output1)                                                                 

#3. 컴파일 훈련

model.compile(loss='categorical_crossentropy', optimizer='adam', # 다중 분류에서는 로스함수를 'categorical_crossentropy' 로 써준다 (99퍼센트로)
              metrics=['accuracy'])

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='auto', verbose=1, 
                              restore_best_weights=True)        

hist = model.fit(x_train, y_train, epochs=1000, batch_size=100,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)

#4. 평가, 예측
results= model.evaluate(x_test, y_test)

y_predict = model.predict(x_test)

y_predict = np.argmax(y_predict, axis= 1)
y_test = np.argmax(y_test, axis= 1)

acc= accuracy_score(y_test, y_predict)
print('loss : ', results[0])
print('acc스코어 : ', acc) 

#================================= [LSTM]loss, accuracy ========================#
# loss :  0.049192022532224655
# acc스코어 :  1.0
#=================================================================================#

#================================= [Conv1D]loss, accuracy ========================#
# loss :  0.054301634430885315
# acc스코어 :  1.0
#=================================================================================#
