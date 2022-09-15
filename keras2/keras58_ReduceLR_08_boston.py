from gc import callbacks
from pickletools import optimize
from random import vonmisesvariate
from tabnanny import verbose
import numpy as np
from keras.datasets import mnist, cifar100
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Input, Dropout
import keras
import tensorflow as tf
print(tf.__version__)from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import MaxAbsScaler,RobustScaler #직접 찾아라!
import numpy as np

datasets = load_boston()
x = datasets.data
y = datasets.target 
#특성 13개

x_train, x_test, y_train, y_test = train_test_split(
    x,  y, train_size= 0.89 , random_state=100
)
# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
print(x_train.shape) #(450, 13)
print(x_test.shape) #(56, 13)

scaler.fit(x_train) #여기까지는 스케일링 작업을 했다.
scaler.transform(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(450, 13,1)
x_test = x_test.reshape(56, 13,1)


# print(np.min(x_train)) # 0.0
# print(np.min(x_test)) # -0.06141956477526944  train 범위에서 없는 데이터가 test에 있는 걸 확인할 수 있다.
# print(np.max(x_test)) # 1.1478180091225068
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Dense,Dropout,Flatten,LSTM,Conv1D
import time
start_time = time.time()
#2. 모델구성
model = Sequential()
# model.add(LSTM(100,input_shape=(13,1)))
model.add(Conv1D(10,2,input_shape=(13,1)))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
# model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))
model.summary()

import datetime
# #3. 컴파일,훈련
filepath = './_ModelCheckPoint/K24/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'
#04d :                  4f : 
from tensorflow.python.keras.callbacks import EarlyStopping,ReduceLROnPlateau

earlyStopping = EarlyStopping(monitor='loss', patience=10, mode='min', 
                              verbose=1,restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss',patience=10,
                              mode='auto',verbose=1,factor=0.5)
# mcp = ModelCheckpoint(monitor='val_loss',mode='auto',verbose=1,
#                       save_best_only=True, 
#                       filepath="".join([filepath,'k24_', date, '_', filename])
#                     )
from tensorflow.python.keras.optimizers import adam_v2
learning_rate = 0.01
optimizer = adam_v2.Adam(lr=learning_rate)
model.compile(loss='mae', optimizer=optimizer,metrics=['acc'])
# "".join은 " "사이에 있는 문자열을 합치겠다는 기능
hist = model.fit(x_train, y_train, epochs=150, batch_size=30, 
                validation_split=0.2,
                verbose=2,callbacks = [earlyStopping,reduce_lr]
                )

#4. 평가,예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)


y_predict = model.predict(x_test)
print(y_test.shape) #(56,)
print(y_predict.shape) #(152, 12, 1)
from sklearn.metrics import accuracy_score, r2_score,accuracy_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 :', r2)
end_time =time.time()-start_time
print("걸린 시간 :",end_time)

################################# CNN
# loss : 2.2761423587799072
# r2스코어 : 0.8636196826801059

################################# LSTM
# loss : 2.403717517852783
# r2스코어 : 0.8671276057624697

################################# Conv1d
# loss : 2.3047733306884766
# r2스코어 : 0.8637253107298841
# 걸린 시간 : 10.709434032440186

################################# LR Reduce
# Epoch 00132: ReduceLROnPlateau 
# reducing learning rate to 3.9062499126885086e-05.
# loss : [2.1607258319854736, 0.0]
# r2스코어 : 0.8843890751875195
# 걸린 시간 : 10.73737359046936
from tensorflow.keras.layers import GlobalAveragePooling2D

#1. 데이터

(x_train, y_train), (x_test, y_test) = load_iris.load_data()

x_train = x_train.reshape(50000, 32*32*3)
x_test = x_test.reshape(10000, 32*32*3)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델
activation = 'relu'
drop = 0.2
optimizer = 'adam'

inputs = Input(shape=(32,32,3), name='input')
x = Conv2D(64, (2, 2), padding='valid',
           activation=activation, name='hidden1')(inputs) # 27, 27, 128
x = Dropout(drop)(x)
# x = Conv2D(64, (2, 2), padding='same',
#            activation=activation, name='hidden2')(x) # 13, 13, 64
# x = Dropout(drop)(x)
x = MaxPool2D()(x)
x = Conv2D(32, (3, 3), padding='valid',
           activation=activation, name='hidden3')(x) # 12, 12, 32
x = Dropout(drop)(x)
# x = Flatten()(x) # 25*25*32 = 20000
x = GlobalAveragePooling2D()(x)
x = Dense(64, activation=activation, name='hidden4')(x)
x = Dropout(drop)(x)
outputs = Dense(100, activation='softmax', name='outputs')(x)    

model = Model(inputs=inputs, outputs=outputs)


# model.summary()


model.compile(optimizer=optimizer, metrics=['acc'], 
                loss='categorical_crossentropy')

import time
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto', verbose=1, factor=0.5)
start = time.time()
model.fit(x_train, y_train, epochs=100, validation_split=0.4, callbacks=[es, reduce_lr], batch_size=128)
end = time.time()

loss, acc = model.evaluate(x_test,y_test)

print("걸린시간 : ", end-start)
# print("model.best_score_ :", model.best_score_)
# print("model.score :", model.score)

from sklearn.metrics import accuracy_score

y_pred = model.predict(x_test)
# print(y_pred[:10])
# y_pred = np.argmax(model.predict(x_test), axis=-1)
print("loss : ", loss)
print("acc : ", acc)
# print("acc : ", accuracy_score(y_test, y_pred))

# acc :  0.9265