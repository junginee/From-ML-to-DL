from warnings import filters
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Activation, Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout
from keras.datasets import mnist, fashion_mnist, cifar10, cifar100
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical # https://wikidocs.net/22647 케라스 원핫인코딩
from sklearn.preprocessing import OneHotEncoder  # https://psystat.tistory.com/136 싸이킷런 원핫인코딩
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from keras.layers import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
oh = OneHotEncoder(sparse=False) # sparse=true 는 매트릭스반환 False는 array 반환
###########################폴더 생성시 현재 파일명으로 자동생성###########################################
import inspect, os
a = inspect.getfile(inspect.currentframe()) #현재 파일이 위치한 경로 + 현재 파일 명
print(a)
print(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))) #현재 파일이 위치한 경로
print(a.split("\\")[-1]) #현재 파일 명
current_name = a.split("\\")[-1]
##########################밑에 filepath경로에 추가로  + current_name + '/' 삽입해야 돌아감###################

x_train = np.load('d:/study_data/_save/_npy/keras49_4_train_x.npy')
y_train = np.load('d:/study_data/_save/_npy/keras49_4_train_y.npy')
x_test = np.load('d:/study_data/_save/_npy/keras49_4_test_x.npy')
y_test = np.load('d:/study_data/_save/_npy/keras49_4_test_y.npy')




#2. 모델구성
model = Sequential()
input1 = Input(shape=(32,32,3))
conv2D_1 = Conv2D(100,3, padding='same')(input1)
MaxP1 = MaxPooling2D()(conv2D_1)
drp1 = Dropout(0.2)(MaxP1)
conv2D_2 = Conv2D(200,2,
                  activation='relu')(drp1)
MaxP2 = MaxPooling2D()(conv2D_2)
drp2 = Dropout(0.2)(MaxP2)
conv2D_3 = Conv2D(200,2, padding='same',
                  activation='relu')(drp2)
MaxP3 = MaxPooling2D()(conv2D_3)
drp3 = Dropout(0.2)(MaxP3)
flatten = Flatten()(drp3)
dense1 = Dense(200)(flatten)
batchnorm1 = BatchNormalization()(dense1)
activ1 = Activation('relu')(batchnorm1)
drp4 = Dropout(0.2)(activ1)
dense2 = Dense(100)(drp4)
batchnorm2 = BatchNormalization()(dense2)
activ2 = Activation('relu')(batchnorm2)
drp5 = Dropout(0.2)(activ2)
dense3 = Dense(100)(drp5)
batchnorm3 = BatchNormalization()(dense3)
activ3 = Activation('relu')(batchnorm3)
drp6 = Dropout(0.2)(activ3)
output1 = Dense(100, activation='softmax')(drp6)
model = Model(inputs=input1, outputs=output1)   


# # (kernel_size * channels +bias) * filters = summary param # (CNN모델)

# x = x.reshape(10,2) 현재 데이터를 순서대로 표기된 행렬로 바꿈

#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M") # 0707_1723
print(date)

save_filepath = './_ModelCheckPoint/' + current_name + '/'
load_filepath = './_ModelCheckPoint/' + current_name + '/'

# model = load_model(load_filepath + '0708_1753_0011-0.0731.hdf5')


filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

earlyStopping = EarlyStopping(monitor='val_loss', patience=10, mode='auto', verbose=1, 
                              restore_best_weights=True)        

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, 
                      filepath= "".join([save_filepath, date, '_', filename])
                      )

hist = model.fit(x_train, y_train, epochs=20, batch_size=32,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test) 
y_predict = model.predict(x_test)


acc = hist.history['accuracy']
val_accuracy = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss : ', loss[-1])
print('accuracy : ', acc[-1])


# loss :  [2.4379518032073975, 0.3725000023841858]
# acc스코어 :  0.3725

# loss :  2.648488759994507
# accuracy :  0.33063751459121704