from warnings import filters
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Activation, Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, accuracy_score
from keras.preprocessing.image import ImageDataGenerator
import time
start = time.time()


#1. 데이터

x_train = np.load('d:/study_data/_save/_npy/keras49_6_train_x.npy')
y_train = np.load('d:/study_data/_save/_npy/keras49_6_train_y.npy')
x_test = np.load('d:/study_data/_save/_npy/keras49_6_test_x.npy')
y_test = np.load('d:/study_data/_save/_npy/keras49_6_test_y.npy')

print(x_train.shape,y_train.shape) #(160, 100, 100, 3) (160,)
print(x_test.shape,y_test.shape)

#2. 모델
model = Sequential()
model.add(Conv2D(150,(2,2), input_shape=(100,100,3), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(100,(3,3), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(100,(3,3), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(100,(3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(160, activation='relu'))
model.add(Dense(2, activation='softmax'))

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

earlyStopping = EarlyStopping(monitor='val_loss', patience=20, mode='auto', verbose=1, 
                              restore_best_weights=True)        


hist = model.fit(x_train, y_train, epochs=50, batch_size=150, validation_split=0.2,
                 callbacks=[earlyStopping])

acc = hist.history['accuracy']
val_accuracy = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss : ', loss[-1])
print('val_loss : ', val_loss[-1])
print('accuracy : ', acc[-1])
print('val_accuracy : ', val_accuracy[-1])


# 4. 평가, 예측
loss = model.evaluate(x_test,y_test)
y_predict = model.predict(x_test)

# acc = accuracy_score(y_test, y_predict)
print('loss: ', loss)
print('acc스코어 : ', acc[-1])
print("time :", time.time() - start)