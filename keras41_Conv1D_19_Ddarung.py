import numpy as np 
import pandas as pd # csv 파일 당겨올 때 사용
from tensorflow.python.keras.models import Sequential,Model,load_model
from tensorflow.python.keras.layers import Dense,Dropout,Input,LSTM,Conv1D,Flatten
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error 
from collections import Counter
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler
import datetime

#1.데이터 
path = './_data/ddarung/'
train_set = pd.read_csv(path + 'train.csv', index_col=0) 
print(train_set)
print(train_set.shape) # (1459, 10)

test_set = pd.read_csv(path + 'test.csv', index_col=0)  

train_set = train_set.dropna()
test_set= test_set.fillna(test_set.mean())

x = train_set.drop(['count'], axis=1,)
y = train_set['count']

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    train_size=0.7,
                                                    shuffle=True,
                                                    random_state=48)

scaler = MaxAbsScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape,x_test.shape)
x_train =x_train.reshape(929,9,1)
x_test = x_test.reshape(399,9,1)

print(x_train.shape, x_test.shape) #(929, 9, 1) (399, 9, 1)


#2. 모델구성

model = Sequential()
model.add(Conv1D(200,2,activation='relu', input_shape=(9,1))) 
model.add(Flatten())
model.add(Dense(100,activation= 'relu'))
model.add(Dense(100,activation= 'relu'))
model.add(Dense(100,activation= 'relu'))
model.add(Dense(100,activation= 'relu'))
model.add(Dense(1))

#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
from tensorflow.python.keras.callbacks import EarlyStopping



earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='auto', verbose=1, 
                              restore_best_weights=True)        



hist = model.fit(x_train, y_train, epochs=1000, batch_size=100,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)

#4. 평가, 예측
loss = model.evaluate(x, y) 


y_predict = model.predict(x_test)

def RMSE(a, b): 
    return np.sqrt(mean_squared_error(a, b))

rmse = RMSE(y_test, y_predict)


from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)

print('loss : ', loss)
print("RMSE : ", rmse)
print('r2스코어 : ', r2)
print('따릉이')

#================================= [LSTM] 결과 ===================================#
# loss :  [1899557.875, 1320.66015625]
# RMSE :  39.6789007900789
# r2스코어 :  0.7753489169625116
#=================================================================================#

#================================= [conv1] 결과 ===================================#

#=================================================================================#


