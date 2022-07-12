import numpy as np
from sklearn import datasets
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.python.keras.layers import Dense, Input, Dropout, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import r2_score, accuracy_score
import time



#1. 데이터
datasets = load_boston()
x, y = datasets.data, datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=66)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape) # (354, 13) (152, 13) (354,) (152,)

scaler = MaxAbsScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(354, 13, 1, 1) 
x_test = x_test.reshape(152, 13, 1, 1)
print(x_train.shape)    # (354, 13, 1, 1)
print(np.unique(x_train, return_counts=True))


#2. 모델 구성
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(1, 1),    
                 padding='same', 
                 input_shape=(13, 1, 1)))              
model.add(Dropout(0.2))
model.add(Conv2D(64, (1, 1), padding='valid', activation='relu'))                         
model.add(Dropout(0.2))
model.add(Conv2D(64, (1, 1), padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(128, (1, 1), padding='valid', activation='relu'))                         
model.add(Dropout(0.2))
model.add(Conv2D(128, (1, 1), padding='same', activation='relu'))
model.add(Dropout(0.2))

model.add(Flatten())    # (N, 63)  (N, 175)
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()


#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam', metrics=['mse']) 

# import datetime
# date = datetime.datetime.now()      
# date = date.strftime("%m%d_%H%M")  
# print(date)

#filepath = './_ModelCheckPoint/k31/'
#filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

earlyStopping = EarlyStopping(monitor='val_loss', patience=20, mode='min',
                              restore_best_weights=True,
                              verbose=1)
#mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, 
                    #   save_best_only=True, 
                    #   filepath="".join([filepath, '01_', date, '_', filename]))
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=500, batch_size=32,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)
end_time = time.time() - start_time


#4. 평가, 예측
loss, acc = model.evaluate(x_test, y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)  
r2 = r2_score(y_test, y_predict)
print('r2 스코어: ', r2)  

#=============================== loss, accuracy ====================================#
# loss :  2.6951115131378174
# r2 스코어:  0.8368823635241539
#===================================================================================#