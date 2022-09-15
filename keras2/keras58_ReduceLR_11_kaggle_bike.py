import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential,load_model
from tensorflow.python.keras.layers import Dense,Dropout,Conv2D,Flatten,LSTM,Conv1D
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
matplotlib.rcParams['font.family']='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus']=False


#1. 데이터
path = 'C:\_data\kaggle_bike/' # ".은 현재 폴더"
train_set = pd.read_csv(path + 'train.csv',
                        index_col=0)
print(train_set)

print(train_set.shape) #(10886, 11)

test_set = pd.read_csv(path + 'test.csv', #예측에서 쓸거야!!
                       index_col=0)

sampleSubmission = pd.read_csv(path + 'sampleSubmission.csv',#예측에서 쓸거야!!
                       index_col=0)
            
print(test_set)
print(test_set.shape) #(6493, 8) #train_set과 열 값이 '1'차이 나는 건 count를 제외했기 때문이다.예측 단계에서 값을 대입

print(train_set.columns)
print(train_set.info()) #null은 누락된 값이라고 하고 "결측치"라고도 한다.
print(train_set.describe()) 


###### 결측치 처리 1.제거##### dropna 사용
print(train_set.isnull().sum()) #각 컬럼당 결측치의 합계
print(train_set.shape) #(10886,11)


x = train_set.drop([ 'casual', 'registered','count'],axis=1) #axis는 컬럼 


print(x.columns)
print(x.shape) #(10886, 8)

y = train_set['count']
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size = 0.949, shuffle = True, random_state = 100
 )
from sklearn.preprocessing import MaxAbsScaler,RobustScaler 
from sklearn.preprocessing import MinMaxScaler,StandardScaler
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train) #여기까지는 스케일링 작업을 했다.
scaler.transform(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(test_set)
# print(y)
# print(y.shape) # (10886,)
print(x_train) #(10330, 8)
print(x_test) #(556, 8)

x_train = x_train.reshape(10330, 8,1)
x_test = x_test.reshape(556, 8,1)



#2. 모델구성
model = Sequential()
# model.add(LSTM(10,input_shape=(8,1)))
model.add(Conv1D(10,2,input_shape=(8,1)))
model.add(Flatten())
model.add(Dense(100,activation='relu'))
# model.add(Dropout(0.3))
model.add(Dense(100, activation='relu'))
# model.add(Dropout(0.3))
model.add(Dense(100, activation='relu'))
# model.add(Dropout(0.3))
model.add(Dense(1))
import datetime
date = datetime.datetime.now()
print(date)

date = date.strftime("%m%d_%H%M") # 0707_1723
print(date)

import time
start_time = time.time()

# #3. 컴파일,훈련
filepath = './_ModelCheckPoint/K24/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
#04d :                  4f : 
from tensorflow.python.keras.callbacks import EarlyStopping,ReduceLROnPlateau

earlyStopping = EarlyStopping(monitor='loss', patience=10, mode='min', 
                              verbose=1,restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss',patience=10,
                              mode='auto',verbose=1,factor=0.5)

from tensorflow.python.keras.optimizers import adam_v2
learning_rate = 0.01
optimizer = adam_v2.Adam(lr=learning_rate)
model.compile(loss='mae', optimizer=optimizer,metrics=['acc'])

hist = model.fit(x_train, y_train, epochs=50, batch_size=2024, 
                validation_split=0.3,
                callbacks = [earlyStopping,reduce_lr],
                verbose=2)


#4. 평가,예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)
y_predict = model.predict(x_test)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 :', r2)
end_time =time.time()-start_time
print("걸린 시간 :",end_time)
# drop 아웃 전 
# loss : 96.13848114013672
# r2스코어 : 0.3580386445346151
# drop 아웃 후
# oss : 100.87238311767578
# r2스코어 : 0.286859120683037


# drop 아웃 전 
# loss : 23656.802734375
# r2스코어 : 0.8237623045901245
# drop 아웃 후
# loss : 30579.8984375
# r2스코어 : 0.739841873958865

#cnn dnn 후
# loss : 94.66157531738281
# r2스코어 : 0.3554885409570203

#######LSTM
# loss : 125.75883483886719
# r2스코어 : 0.0633013409788773
#######Conv1d
# loss : 100.57544708251953
# r2스코어 : 0.3071249407330422
# 걸린 시간 : 4.462159633636475
#######Conv1d +LR Reduce
# loss : [94.62430572509766, 0.016187049448490143]
# r2스코어 : 0.34943541194259764
# 걸린 시간 : 3.9924404621124268
