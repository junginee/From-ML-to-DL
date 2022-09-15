from keras.models import Sequential, Model, load_model
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Conv2D, Flatten, GlobalAveragePooling2D
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import validation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)         
# print(x_train.shape, y_train.shape)    
# print(x_test.shape, y_test.shape)      

x_train = x_train.reshape(60000, 28, 28, 1) 
x_test = x_test.reshape(10000, 28, 28, 1)
                                          
                                          
#10개 array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


# print(y)
# print(y.shape)
print(x_train.shape, y_train.shape) #(60000, 28, 28, 1) (60000,)


# 2. 모델구성

model  =  Sequential() 
model.add(Conv2D(10, kernel_size=(2,2), input_shape=(28, 28, 1 ) ))
model.add(Conv2D(5, (2,2), activation="relu") )
model.add(Dropout(0.2))
model.add(Conv2D(7, (2,2), activation="relu") )

model.add(GlobalAveragePooling2D())
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(32, activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(10, activation='softmax'))

#3. 컴파일, 훈련
optimizer = 'adam'
model.compile(optimizer=optimizer, metrics=['acc'], 
                                loss='sparse_categorical_crossentropy')

import time
from tensorflow.python.keras.callbacks import TensorBoard,EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto', verbose=1, factor=0.5)

tb = TensorBoard(log_dir='D:\study_data\\tensorboard_log\_graph', 
                            histogram_freq=0, write_graph=True, write_images=True)

# 실행방법 : tensorboard --logdir=. (경로)
# http://localhost:6006/ (텐서보드 로컬호스트)
# http://127.0.01:6006   (127.0.07 = 내 pc 고유번호)

start = time.time()
hist=model.fit(x_train, y_train, epochs=100, validation_split=0.4, callbacks=[es, reduce_lr, tb]
                 ,batch_size=128) 
end = time.time() - start

#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)

print('loss : ', round(loss,4))
print('accuracy : ', round(acc,4))
print('걸린시간 :', round(start-end,4))

################EDA###############

import matplotlib.pyplot as plt

# 1
plt.subplot(2, 1, 1)
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend(loc='upper right')

#2
plt.subplot(2, 1, 2)
plt.plot(hist.history['acc'], marker='.', c='red', label='acc')
plt.plot(hist.history['val_acc'], marker='.', c='blue', label='val_acc')
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epochs')
plt.legend(['acc','val_acc'])

plt.show()