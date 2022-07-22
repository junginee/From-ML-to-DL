import pandas as pd
import numpy as np
from warnings import filters
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Activation, Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.image import ImageDataGenerator

x_train = np.load('d:/study_data/_save/_npy/keras49_5_train_x.npy')
y_train = np.load('d:/study_data/_save/_npy/keras49_5_train_y.npy')
x_test = np.load('d:/study_data/_save/_npy/keras49_5_test_x.npy')
y_test = np.load('d:/study_data/_save/_npy/keras49_5_test_y.npy')

#2. 모델
model = Sequential()
model.add(Conv2D(10,(2,2), input_shape=(150,150,1), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(10,(3,3), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(10,(3,3), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(10,(3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='auto', verbose=1, 
                              restore_best_weights=True)        

import time
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=200, batch_size=32,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)
end_time = time.time() - start_time

#4. 평가, 예측
loss = model.evaluate(x_test, y_test) 
y_predict = model.predict(x_test)
print(y_test, y_predict)

accuracy = hist.history['accuracy']
val_accuracy = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']
# acc = accuracy_score(y_test, y_predict)

print('loss :', loss[-1])       
print('val_loss :', val_loss[-1])
print('accuracy :', accuracy[-1])
print('val_accuracy :', val_accuracy[-1])

print("=====================================================================")
print("걸린시간 : ", end_time)


# loss : 0.00022540295321960002
# val_loss : 0.31785500049591064
# accuracy : 1.0
# val_accuracy : 0.8999999761581421
# =====================================================================
# 걸린시간 :  10.726546049118042