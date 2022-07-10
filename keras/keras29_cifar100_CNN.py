from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout # 이미지 작업은 2D
from keras.datasets import mnist, cifar100
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


#1. 데이터

(x_train, y_train), (x_test, y_test) = cifar100.load_data()

print(x_train.shape, y_train.shape) # (50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)   # (10000, 32, 32, 3) (10000, 1)

# from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
# scaler = StandardScaler()
# scaler.fit(x_train)
# scaler.transform(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

x_train = x_train.reshape(50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32, 32, 3)
print(x_train.shape) # (50000, 32, 32, 3)

print(np.unique(y_train, return_counts=True))


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


#2. 모델구성
model = Sequential()
model.add(Conv2D(10, kernel_size=(3,3), strides=2, padding='same', input_shape=(32, 32, 3)))   
model.add(MaxPool2D())
model.add(Dropout(0.2))
model.add(Conv2D(5, (2,2), activation='relu')) 
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(100, activation='softmax'))
# model.summary()



#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint


earlyStopping =EarlyStopping(monitor='val_loss', patience=1, mode='min', verbose=1, 
                             restore_best_weights=True) 

#mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, 
#                      save_best_only=True)

hist = model.fit(x_train, y_train, epochs=1, batch_size=1, 
                validation_split=0.2,
                callbacks=[earlyStopping, mcp], 
                verbose=1)



#4. 평가, 예측
loss = model.evaluate(x_test, y_test) 
print('loss : ', loss)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score, accuracy_score
y_predict = np.argmax(y_predict, axis= 1)
y_predict = to_categorical(y_predict)

acc = accuracy_score(y_test, y_predict)
print('acc스코어 : ', acc)

