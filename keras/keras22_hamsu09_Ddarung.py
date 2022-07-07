import numpy as np
import pandas as pd #read_csv, columns, info, describe, 결측치 제공
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import time

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

###############스캘러 방법#####################################
scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)
test_set = scaler.transform(test_set)
print(np.min(x_train))  # 0.0
print(np.max(x_train))  # 1.0

print(np.min(x_test))  # 1.0
print(np.max(x_test))  # 1.0

#2. 모델구성
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input

#함수형 모델
input1 = Input(shape=(9,))
dense1 = Dense(5)(input1)
dense2 = Dense(6)(dense1)
dense3 = Dense(7, activation = 'relu')(dense2)
dense4 = Dense(5)(dense3)
dense5= Dense(8)(dense4)
dense6 = Dense(10)(dense5)
dense7 = Dense(10, activation = 'relu')(dense6)
output1 = Dense(1)(dense7)

model = Model(inputs=input1, outputs=output1)

#Sequential 모델
# model = Sequential()
# model.add(Dense(5,input_dim=9)) 
# model.add(Dense(6))
# model.add(Dense(7, activation='relu'))
# model.add(Dense(5))
# model.add(Dense(8))
# model.add(Dense(10))
# model.add(Dense(10,activation='relu' ))
# model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=500, mode='min', verbose=1, 
                              restore_best_weights=True)

start_time = time.time()
model.fit(x_train, y_train, epochs=600, batch_size=28, verbose=1, validation_split=0.2, callbacks=[earlyStopping])  

end_time = time.time() 

#4. 평가, 예측
print("걸린시간 : ", end_time)

loss = model.evaluate(x_test, y_test)   
print('loss : ', loss)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE :", rmse)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict) 
print('r2스코어 : ', r2)

#[결과]
# 걸린시간 :  1657162401.0372148
# loss :  [1728.5408935546875, 30.45831298828125]
# RMSE : 41.57572361853923
# r2스코어 :  0.7213146563813365


