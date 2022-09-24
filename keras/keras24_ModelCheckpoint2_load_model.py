import numpy as np 
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
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

start_time = time.time()

#2.모델구성

# model = Sequential()
# model.add(Dense(64, input_dim=13)) 
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(1))
# model.summary()



#3. 컴파일, 훈련

# from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
# earlyStopping = EarlyStopping(monitor='loss', patience=200, mode='min', 
#                               verbose=1,restore_best_weights=True)

# model.compile(loss='mse', optimizer='adam')

# mcp = ModelCheckpoint(monitor='val', mode = 'auto', verbose=1,
#                       save_best_only=True,
#                       filepath='./_ModelCheckpoint/keras24_ModelCheckpoint.hdf5'
#                       )


# hist = model.fit(x_train, y_train, epochs=10, batch_size=1,
#                  validation_split=0.2,
#                  callbacks = [earlyStopping, mcp],
#                  verbose=1)

# end_time = time.time() - start_time


model = load_model('./_ModelCheckpoint/keras24_ModelCheckpoint.hdf5')

# 4. 평가, 예측

loss = model.evaluate(x_test, y_test)

print('loss : ' , loss)
y_predict = model.predict(x_test) 
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict) 
print('r2스코어 : ' , r2) 


# loss :  15.001058578491211
# r2스코어 :  0.818426734081434
