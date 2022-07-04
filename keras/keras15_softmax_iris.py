
import numpy as np
from sklearn.datasets import load_iris #1. 데이터
from sklearn.model_selection import train_test_split #1. 데이터

from tensorflow.python.keras.models import Sequential #2. 모델구성
from tensorflow.python.keras.layers import Dense #2. 모델구성

from sklearn.metrics import accuracy_score #3,4  metrics로 accuracy 지표 사용


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


#2. 모델구성

model = Sequential()
model.add(Dense(5,input_dim = 4))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))  

# 열 4개 (sepal length / sepal width / petal length / petal width) 를 Iris-Setosa / Iris-Versicolour / Iris-Virginica로 분류한다.
# 다중분류일 때는 최종 노드의 갯수는 y의 라벨의 갯수 즉, y 값의 종류 >> 마지막 노드 수 맞춰주기
# softmax는 분류 값에 대한 숫자만큼을 노드로 빼준다. 각각을 %로 매기며 총합은 1이다. %중 가장 큰 값을 찾는것이다.
# softmax의 합은 1 / 즉, Iris-Setosa / Iris-Versicolour / Iris-Virginica 중 가장 큰 %를 찾는다.

#오류 발생 :  ValueError: Shapes (None, 1) and (None, 3) are incompatible
# y는 (150, ) 형태이며 이를 (150, 3)로 바꿔줘야 한다. by one hot encoding
# one hot encoding이란? 자연어를 컴퓨터가 처리하도록 하기 위해서 숫자로 바꾸는 방법인 임베딩 중 하나의 방법(가장 기본적인 표현 방법)
# how? by 사이킷런/ by 케라스
  # 1) 케라스 )) from keras.utils import to_categorical
#주의 : 데이터 분류 시 셔플 주의!!  shuffle=True
  


#3. 컴파일, 훈련

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 이진분류모델에서는 loss로 binary_crossentropy
# 다중분류모델에서는 loss로 categorical_crossentropy

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=300, mode='auto', verbose=1, 
                              restore_best_weights=True)        


hist = model.fit(x_train, y_train, epochs=500, batch_size=50,
                 validation_split=0.2,
                 callbacks=[earlyStopping],verbose=1)



#4. 예측, 평가

#[loss, acc 출력방법 1]
loss, acc = model.evaluate(x_test, y_test)
print('loss : ' , loss)
print('accuracy : ', acc) 

#[loss, acc 출력방법 2]
results = model.evaluate(x_test, y_test)
print('loss : ' , results[0])
print('accuracy : ', results[1]) 

y_predict = model.predict(x_test)
acc= accuracy_score(y_test, y_predict)

# y_predict = y_predict.round(0)
# print(y_predict)

