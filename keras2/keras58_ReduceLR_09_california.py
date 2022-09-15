from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Dense,Dropout,Conv2D,Flatten,LSTM,Conv1D
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from tensorflow.python.keras.callbacks import EarlyStopping,ModelCheckpoint
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['font.family']='Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus']=False
#1. 데이터
datasets = fetch_california_housing()
x = datasets.data #데이터를 리스트 형태로 불러올 때 함
y = datasets.target
x_train, x_test ,y_train, y_test = train_test_split(
          x, y, train_size=0.8,shuffle=True,random_state=100)
from sklearn.preprocessing import MaxAbsScaler,RobustScaler 
from sklearn.preprocessing import MinMaxScaler,StandardScaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

scaler.fit(x_train) #여기까지는 스케일링 작업을 했다.
scaler.transform(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
# print(x.shape, y.shape) #(506, 13)-> 13개의 피쳐 (506,) 
print(x_train.shape) #(16512, 8)
print(x_test.shape) #(16512, 8)

x_train = x_train.reshape(16512, 8,1)
x_test = x_test.reshape(4128, 8,1)
# print(datasets.feature_names)
# print(datasets.DESCR)


#2. 모델구성
model = Sequential()
# model.add(LSTM(10,input_shape=(8,1)))
model.add(Conv1D(10,2,input_shape=(8,1)))
model.add(Flatten())
model.add(Dense(100,activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1))
import datetime
date = datetime.datetime.now()
print(date)

date = date.strftime("%m%d_%H%M") # 0707_1723
print(date)
import time
start_time = time.time()

# #3. 컴파일,훈련
# filepath = './_ModelCheckPoint/K24/'
# filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
#04d :                  4f : 
from tensorflow.python.keras.callbacks import EarlyStopping,ReduceLROnPlateau

earlyStopping = EarlyStopping(monitor='loss', patience=10, mode='min', 
                              verbose=1,restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss',patience=10,
                              mode='auto',verbose=1,factor=0.5)

# mcp = ModelCheckpoint(monitor='val_loss',mode='auto',verbose=1,
#                       save_best_only=True, 
                      # filepath="".join([filepath,'k25_', date, '_california_', filename])
                    # )
from tensorflow.python.keras.optimizers import adam_v2
learning_rate = 0.01
optimizer = adam_v2.Adam(lr=learning_rate)
model.compile(loss='mae', optimizer=optimizer,metrics=['acc'])

hist = model.fit(x_train, y_train, epochs=150, batch_size=3080, 
                validation_split=0.3,
                callbacks = [earlyStopping,reduce_lr],
                verbose=2
                )



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
# loss : 0.32516416907310486
# r2스코어 : 0.8211444261300973
# drop 아웃 후
# oss : 0.34795650839805603
# r2스코어 : 0.7869890552305074

# CNN 

# loss : 0.3301199674606323
# r2스코어 : 0.8126296639125675

# LSTM

# loss : 0.3338578939437866
# r2스코어 : 0.8108092924835244

# Conv1d

# loss : 0.34012356400489807
# r2스코어 : 0.792605037820072
# 걸린 시간 : 9.529327154159546

################################# LR Reduce
# Epoch 00124: 
#   ReduceLROnPlateau reducing learning rate to 0.0012499999720603228.
# loss : [0.32297569513320923, 0.0026647287886589766]
# r2스코어 : 0.8138822609681217
# 걸린 시간 : 7.937630414962768
Footer
© 2022 GitHub, Inc.
Footer navigation
Terms
Privacy
Security
Status
Docs
Contact GitHub
Pricing
API
Traini