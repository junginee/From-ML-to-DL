import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sqlalchemy import false
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/gulim.TTc"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
from tensorflow.keras.utils import to_categorical # https://wikidocs.net/22647 케라스 원핫인코딩
from sklearn.preprocessing import OneHotEncoder  # https://psystat.tistory.com/136 싸이킷런 원핫인코딩
from sklearn.datasets import fetch_covtype
import tensorflow as tf

#1. 데이터

datasets = fetch_covtype()
x = datasets.data
y = datasets.target
print(x.shape, y.shape) # (581012, 54) (581012,)
print(np.unique(y)) # [1 2 3 4 5 6 7]
# print(x,y)

# print(datasets.DESCR)
# print(datasets.feature_names)
# print(datasets)

# print(x)
# print(y)
####################케라스########################
# y = to_categorical(y)
# print(np.unique(y, return_counts=True)) # y의 라벨값 :  [1 2 3 4 5 6 7]
#################################################

####################겟더미#######################
# y = pd.get_dummies(y)  #겟더미는 y_predict 할때 np아니고 tf.argmax로 바꾸기
# print(y)
################################################

####################원핫인코더###################
df = pd.DataFrame(y)
print(df)
oh = OneHotEncoder(sparse=False) # sparse=true 는 매트릭스반환 False는 array 반환
y = oh.fit_transform(df)
print(y)
################################################


x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.7,
                                                    random_state=66
                                                    )

print(y_test)
print(y_train)
print(y)


#2. 모델

model = Sequential()
model.add(Dense(30, input_dim=54, activation='relu')) 
model.add(Dense(20, activation='sigmoid'))               
model.add(Dense(20, activation='relu'))            
model.add(Dense(20, activation='relu'))               
model.add(Dense(7, activation='softmax'))          
                                                               

#3. 컴파일 훈련

model.compile(loss='categorical_crossentropy', optimizer='adam', # 다중 분류에서는 로스함수를 'categorical_crossentropy' 로 써준다 (99퍼센트로)
              metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience=50, mode='auto', verbose=1, 
                              restore_best_weights=True)   

model.fit(x_train, y_train, epochs=50, batch_size=5,
                 validation_split=0.2,
                 callbacks=[es],
                 verbose=1)

#4. 평가, 예측
# loss, acc= model.evaluate(x_test, y_test)
# print('loss : ', loss)
# print('accuracy : ', acc)

results= model.evaluate(x_test, y_test)
print('loss : ', results[0])
print('accuracy : ', results[1])


y_predict = model.predict(x_test)
print(y_predict)
print(y_test)
y_predict = np.argmax(y_predict, axis= 1)  # 판다스 겟더미 쓸때는 tf.argmax sklearn 원핫인코딩 쓸때는 np
print(y_predict)
y_test = np.argmax(y_test, axis= 1)
print(y_test)
# y_predict = to_categorical(y_predict)
# y_test = np.argmax(y_test, axis= 1)
print(np.unique(y_predict))
print(np.unique(y_test))



acc= accuracy_score(y_test, y_predict)
print('acc스코어 : ', acc) 

#drop out 이전
# loss :  0.9041478037834167
# accuracy :  0.6589177250862122

#drop out 이후
