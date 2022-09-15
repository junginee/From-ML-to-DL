import numpy as np
from tensorflow.keras.datasets import mnist, cifar100
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dropout, Dense, Conv2D, Flatten, MaxPool2D, Input,  MaxPooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D


#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar100.load_data()

x_train = x_train.reshape(50000, 32*32*3)   
x_test = x_test.reshape(10000, 32*32* 3)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(50000, 32,32,3)
x_test = x_test.reshape(10000, 32,32,3)


from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#2. 모델
activation = 'relu'
drop = 0.2

inputs = Input(shape=(32,32,3), name='input')
x = Conv2D(128, (2, 2), activation=activation, padding='valid', name='hidden1')(inputs)
x = Dropout(drop)(x)
x = MaxPooling2D()(x)
x = Conv2D(32, (3, 3), activation=activation, padding='valid', name='hidden3')(x)
x = Dropout(drop)(x) 

# x = Flatten()(x) # (25*25*32) / Flatten의 문제점: 연산량이 너무 많아짐
x = GlobalAveragePooling2D()(x)

x = Dense(256, activation=activation, name='hidden4')(x)
x = Dropout(drop)(x)
x = Dense(128, activation=activation, name='hidden5')(x)
x = Dropout(drop)(x)
outputs = Dense(100, activation='softmax', name='outputs')(x)

model = Model(inputs=inputs, outputs=outputs)

model.summary()   

#3. 컴파일, 훈련
optimizer = 'adam'
model.compile(optimizer=optimizer, metrics=['acc'], 
                                loss='categorical_crossentropy')

import time
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto', verbose=1, factor=0.5)

start = time.time()
model.fit(x_train, y_train, epochs=100, validation_split=0.4, callbacks=[es, reduce_lr]
                 ,batch_size=128) 
end = time.time() - start

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)

print('loss : ', round(loss,4))
print('accuracy : ', round(acc,4))
print('걸린시간 :', round(start-end,4))