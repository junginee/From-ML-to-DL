#데이콘 따릉이 문제풀이

#모든 훈련, 평가는 train set으로 한다. test set에는 count가 빠져있는데, train set으로 훈련시킨 후 test set의 count 값 715개를 예측해서 제출해라
#train set에 10개의 컬럼이 있는데 9개 컬럼은 x변수로 하고, 1개(count) 컬럼은 y 값에 집어 넣는다. 이 count 값을 test set의 결과로 제출할 것
#그리고 이 10개의 컬럼을 가지고 train70%, test30% 으로 설정한다.
#원래 train set에는 총11개의 컬럼이 있는데 id는 index로 분류하였기에 10개의 컬럼으로 (x,y) 트레인, 테스트 한다.

#문제해결하기
#모델 구성 후 실행했을 때 nan 값이 나왔다.
#해결방법은? nan 값 들어간 행을 제외시킨다. 결측치가 들어있는 행을 삭제함으로써 이 방법을 해결할 수 있으며 데이터가 많을 경우 이 단순작업은 위험한 방법이다. 
#따라서 nan 값 해결하는 방법은 추후 배울 예정

import numpy as np
import pandas as pd #read_csv, columns, info, describe, 결측치 제공
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#1.데이터
path = './_data/ddarung/'
train_set = pd.read_csv(path + 'train.csv', index_col =0) #id는 0번째에 위치한다. #[1459 rows x 10 columns]

print(train_set)
print(train_set.shape) #(1459,10)

test_set = pd.read_csv(path + 'test.csv', index_col =0) #예측에서 쓸것이다.
print(test_set)
print(test_set.shape)  #(715, 9)

print(train_set.columns)
print(train_set.info())  #결측치 : 데이터가 빠진 ..
print(train_set.describe())


#### 결측치 처리 1. 제거 ####
print(train_set.isnull().sum()) #train set에 있는 널값의 합계를 구한다.
train_set = train_set.dropna() #결측치가 들어있는 행을 삭제한다.
print(train_set.isnull().sum()) #결측치 제거 후 train set에 들어있는 널값의 합계를 구한다.
############################
x = train_set.drop(['count'], axis = 1) #x 변수에는 count 열을 제외한 나머지 컬럼을 저장한다.

print(x)
print(x.columns)
print(x.shape) #(1459,9)

y = train_set['count'] #count 컬럼만 y 변수에 저장한다.
print(y)
print(y.shape)

x_train,x_test,y_train, y_test = train_test_split(x,y, train_size=0.7, shuffle=True, random_state=72)

#2. 모델구성
model = Sequential()
model.add(Dense(5,input_dim=9)) 
model.add(Dense(6))
model.add(Dense(7))
model.add(Dense(5))
model.add(Dense(8))
model.add(Dense(10))
model.add(Dense(10))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train,y_train, epochs=600, batch_size=28)  

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)   
print('loss : ', loss)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE :", rmse)
# y_predict = model.predict(test_set) #y_predict 값이 제출할 값이다.

 

       
