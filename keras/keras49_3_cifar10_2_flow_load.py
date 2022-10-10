from warnings import filters
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Activation, Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout
from keras.datasets import mnist
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from keras.preprocessing.image import ImageDataGenerator


x_train = np.load('d:/study_data/_save/_npy/keras49_3_train_x.npy')
y_train = np.load('d:/study_data/_save/_npy/keras49_3_train_y.npy')
x_test = np.load('d:/study_data/_save/_npy/keras49_3_test_x.npy')
y_test = np.load('d:/study_data/_save/_npy/keras49_3_test_y.npy')


#2. 모델구성
input1 = Input(shape=(32, 32, 3 ))
dense1 = Conv2D(100,3, padding='same')(input1)
dense2 = MaxPooling2D()(dense1)
dense3 = Dropout(0.2)(dense2)
dense4 = Conv2D(7, (2,2), activation="relu")(dense3)
dense4 = Flatten()(dense4)
dense5 = Dense(64, activation="relu")(dense4)
dense6 = Dropout(0.2)(dense5)
dense7 = Dense(32, activation="relu")(dense6)
dense8 = Dense(16, activation="relu")(dense7)
output1 = Dense(10,  activation='softmax')(dense8)

model = Model(inputs=input1, outputs=output1)


#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
from tensorflow.python.keras.callbacks import EarlyStopping

earlyStopping = EarlyStopping(monitor='val_loss', patience=50, mode='auto', verbose=1, 
                              restore_best_weights=True)        


hist = model.fit(x_train, y_train, epochs=300, batch_size=32,
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

# loss :  0.44089275598526
# accuracy :  0.9411764740943909
