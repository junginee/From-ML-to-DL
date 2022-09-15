#trainable = true, false 비교해가면서 만들어서 결과 비교 (가중치 동결, 비동결)
import time
import numpy as np
import tensorflow as tf
from keras.datasets import cifar10
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense,Conv2D,Flatten,Dropout
from keras.applications import VGG16
from sklearn.metrics import accuracy_score
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau

#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

#2. 모델
vgg16 = VGG16(weights='imagenet', include_top=False,
               input_shape=(32,32,3))
vgg16.trainable=False 
model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(10))
model.add(Dense(1))
vgg16.trainable=False 

activation = 'relu'
drop = 0.2
optimizer = 'adam'
model.compile(optimizer=optimizer, metrics=['acc'], 
                loss='categorical_crossentropy')


es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto', verbose=1, factor=0.5)
start = time.time()
model.fit(x_train, y_train, epochs=100, validation_split=0.4, callbacks=[es, reduce_lr], batch_size=128)
end = time.time()

loss, acc = model.evaluate(x_test,y_test)
print("걸린시간 : ", end-start)

y_pred = model.predict(x_test)
print("loss : ", loss)
print("acc : ", acc)

#vgg16.trainable=True/ model=true 
# loss :  5.364418598219345e-07
# acc :  0.10999999940395355

#vgg16.trainable=False / model= true 
# loss :  5.364418598219345e-07
# acc :  0.11599999666213989

#vgg16.trainable=False  / model false
# loss :  5.364418598219345e-07
# acc :  0.1103999987244606