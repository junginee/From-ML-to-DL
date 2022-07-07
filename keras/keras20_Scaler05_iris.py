#[과제] --- 완료
#1. scaler 하기 전
# loss :  0.26301804184913635
# accuracy :  0.9333333373069763

#2. MinMaxScaler()
# loss :  0.35637184977531433
# accuracy :  1.0

#3. StandardScaler()
# loss :  0.10914774239063263
# accuracy :  0.9666666388511658

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.datasets import load_iris #1. 데이터
from sklearn.model_selection import train_test_split #1. 데이터

from tensorflow.python.keras.models import Sequential #2. 모델구성
from tensorflow.python.keras.layers import Dense #2. 모델구성

from sklearn.metrics import accuracy_score #3,4  metrics로 accuracy 지표 사용
import time

import tensorflow as tf
tf.random.set_seed(66)

#1. 데이터

datasets = load_iris()
# print(datasets.DESCR)
# print(datasets.feature_names)

x = datasets['data']
y = datasets.target

print(x) #(150, 4) # 150행 : Number of Instances: 150 (50 in each of three classes :  Iris-Setosa / Iris-Versicolour / Iris-Virginica)
                   # 4열 : sepal length / sepal width / petal length / petal width
print(y) #(150, )
print(x.shape, y.shape) #(150, 4) #(150, )
print("y의 라벨값(y의 고유값)", np.unique(y)) #y의 라벨값(y의 고유값) [0 1 2]

from tensorflow.keras.utils import to_categorical 
y = to_categorical(y)

print(y)
print(y.shape) #(150,3)

x_train, x_test, y_train, y_test = train_test_split( x, y, train_size = 0.8, shuffle=True, random_state=68 )


###############스캘러 방법 2가지###############################
#scaler = StandardScaler()
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)
print(np.min(x_train)) #0.0 
print(np.max(x_train)) #1.0

#2. 모델구성

model = Sequential()
model.add(Dense(5,input_dim = 4))
model.add(Dense(10, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(3, activation='softmax'))  

# 열 4개 (sepal length / sepal width / petal length / petal width) 를 Iris-Setosa / Iris-Versicolour / Iris-Virginica로 분류한다.
# 다중분류일 때는 최종 노드의 갯수는 y의 라벨의 갯수 즉, y 값의 종류 >> 마지막 노드 수 맞춰주기 
# model.add(Dense(3, activation='softmax')) 다중분류의 softmax는 마지막 활성화함수에만 준다. (중간에 주면 터짐!)
# softmax는 분류 값에 대한 숫자만큼을 노드로 빼준다. 각각을 %로 매기며 총합은 1이다. %중 가장 큰 값을 찾는것이다.
# softmax의 합은 1 이며, Iris-Setosa / Iris-Versicolour / Iris-Virginica 중 가장 큰 %를 찾는다.

#오류 발생 :  ValueError: Shapes (None, 1) and (None, 3) are incompatible
# y는 (150, ) 형태이며 이를 (150, 3)로 바꿔줘야 한다. by one hot encoding
# one hot encoding이란? 자연어를 컴퓨터가 처리하도록 하기 위해서 숫자로 바꾸는 방법인 임베딩 중 하나의 방법(가장 기본적인 표현 방법)
# how? # how? by 사이킷런/ by 케라스 / by 판다스

#주의 : 데이터 분류 시 셔플 주의!!  shuffle=True
  


#3. 컴파일, 훈련

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 이진분류모델에서는 loss로 binary_crossentropy
# 다중분류모델에서는 loss로 categorical_crossentropy

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=300, mode='auto', verbose=1, 
                              restore_best_weights=True)        


hist = model.fit(x_train, y_train, epochs=100, batch_size=50,
                 validation_split=0.2,
                 callbacks=[earlyStopping],verbose=1)



#4. 평가, 예측


#[loss, acc 출력방법 1]
loss, acc = model.evaluate(x_test, y_test)
print('loss : ' , loss)
print('accuracy : ', acc) 

#[loss, acc 출력방법 2]
results = model.evaluate(x_test, y_test)
print('loss : ' , results[0])
print('accuracy : ', results[1]) 

print("----------------------------------------")

from sklearn.metrics import r2_score, accuracy_score  
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict, axis= 1)
print(y_predict)

y_test = np.argmax(y_test, axis= 1)
print(y_test)

acc= accuracy_score(y_test, y_predict)
print('acc스코어 : ', acc) 

# [결과]
# loss :  0.2630181908607483
# accuracy :  0.9333333373069763
# ----------------------------------------
# [2 0 0 1 1 1 1 0 1 0 0 0 2 2 1 1 1 2 2 2 2 1 0 2 1 0 2 0 2 1]
# [2 0 0 1 1 1 1 0 1 0 0 0 2 2 1 1 2 2 2 2 2 1 0 2 1 0 2 0 2 2]
# acc스코어 :  0.9333333333333333