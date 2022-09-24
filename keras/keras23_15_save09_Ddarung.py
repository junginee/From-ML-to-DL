import numpy as np
import pandas as pd 
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import time

#1.데이터
path = './_data/ddarung/'
train_set = pd.read_csv(path + 'train.csv', index_col =0) 

print(train_set)
print(train_set.shape) 

test_set = pd.read_csv(path + 'test.csv', index_col =0)
print(test_set)
print(test_set.shape) 

print(train_set.columns)
print(train_set.info())  
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


scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)
test_set = scaler.transform(test_set)
print(np.min(x_train))  # 0.0
print(np.max(x_train))  # 1.0

print(np.min(x_test))  # 1.0
print(np.max(x_test))  # 1.0

#2. 모델구성
model = Sequential()
model.add(Dense(5,input_dim=9)) 
model.add(Dense(6))
model.add(Dense(7, activation='relu'))
model.add(Dense(5))
model.add(Dense(8))
model.add(Dense(10))
model.add(Dense(10,activation='relu' ))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=500, mode='min', verbose=1, 
                              restore_best_weights=True)

start_time = time.time()
model.fit(x_train, y_train, epochs=600, batch_size=28, verbose=1, validation_split=0.2, callbacks=[earlyStopping])  

end_time = time.time() 


model.save_weights("./_save/keras23_15_save09_Ddarung.h5")

#4. 평가, 예측
print("걸린시간 : ", end_time)

loss = model.evaluate(x_test, y_test)   
print('loss : ', loss)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE :", rmse)
print('r2스코어 : ', r2)
