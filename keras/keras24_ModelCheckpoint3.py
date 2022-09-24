import numpy as np 
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
import time   

#1.데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.7, random_state = 66)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train) 
x_test = scaler.transform(x_test)



#2.모델구성

model = Sequential()
model.add(Dense(64, input_dim=13)) 
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))
model.summary()



#3. 컴파일, 훈련

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
earlyStopping = EarlyStopping(monitor='loss', patience=200, mode='min', 
                              verbose=1,restore_best_weights=True)

model.compile(loss='mse', optimizer='adam')

mcp = ModelCheckpoint(monitor='val_loss', mode = 'auto', verbose=1,
                      save_best_only=True,
                      filepath='./_ModelCheckpoint/keras24_ModelCheckpoint3.hdf5'
                      )

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=10, batch_size=10,
                 validation_split=0.2,
                 callbacks = [earlyStopping, mcp],
                 verbose=1)

end_time = time.time() - start_time


model.save('./_save/keras24_3_save_model.h5')

# 4. 평가, 예측

loss = model.evaluate(x_test, y_test)

print('loss : ' , loss)
y_predict = model.predict(x_test) 

r2 = r2_score(y_test, y_predict) 
print('r2스코어 : ' , r2) 


print("======================== 1. 기본 출력 ========================")
loss = model.evaluate(x_test, y_test)
print('loss : ', loss)
y_predict = model.predict( x_test )
r2 = r2_score(y_test, y_predict) #y-predict test비교
print('r2스코어 : ', r2)

print("======================== 2. load_model 출력 ========================")
model2 = load_model('./_save/keras24_3_save_model.h5')
loss2 = model2.evaluate(x_test, y_test)
y_predict = model.predict( x_test )
r2 = r2_score(y_test, y_predict) #y-predict test비교
print('r2스코어 : ', r2)


print("======================== 3. ModelCheckPoint 출력 ========================")
model3 = load_model('./_ModelCheckpoint/keras24_ModelCheckpoint3.hdf5')
loss3 = model3.evaluate(x_test, y_test)
print('loss : ', loss3)
y_predict = model.predict( x_test )
r2 = r2_score(y_test, y_predict) #y-predict test비교      
print('r2스코어 : ', r2)
