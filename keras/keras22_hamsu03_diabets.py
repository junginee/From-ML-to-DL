import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import time


#1. 데이터
datasets =load_diabetes()
x = datasets.data
y = datasets.target
x_train,x_test,y_train, y_test = train_test_split(x,y, train_size=0.75, shuffle=True, random_state=72)


scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)
print(np.min(x_train)) #0.0 
print(np.max(x_train)) #1.0

# print(x)
# print(y) 
# print(x.shape, y.shape) #(442, 10) (442,)
# print(datasets.feature_names)
# print(datasets.DESCR)


#2.모델구성
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input


#함수형 모델
input1 = Input(shape=(10,))
dense1 = Dense(6)(input1)
dense2 = Dense(15, activation = 'sigmoid')(dense1)
dense3 = Dense(15, activation = 'relu')(dense2)
dense4 = Dense(20)(dense3)
dense5= Dense(20, activation = 'relu')(dense4)
dense6 = Dense(20)(dense5)
dense7 = Dense(50, activation = 'sigmoid')(dense6)
output1 = Dense(1)(dense7)


model = Model(inputs=input1, outputs=output1)

#Sequential 모델
# model = Sequential()
# model.add(Dense(6,input_dim=10))
# model.add(Dense(15, activation='sigmoid'))
# model.add(Dense(15, activation='relu'))
# model.add(Dense(20))
# model.add(Dense(20, activation='relu'))
# model.add(Dense(20))
# model.add(Dense(50, activation='sigmoid'))
# model.add(Dense(1))


#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam',
              metrics=['accuracy'])
        
start_time = time.time()
      
from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping =EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1, 
                             restore_best_weights=True) 
                             
end_time = time.time()

hist = model.fit(x_train, y_train, epochs=300, batch_size=5, 
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)

#4.평가, 예측
print("걸린시간 : ", end_time)


loss = model.evaluate(x_test, y_test)
print('loss : ',loss)


y_predict = model.predict(x_test) 


from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict) #R 제곱 = 예측값 (y_predict) / 실제값 (y_test) 
print('r2스코어 : ', r2)

#[결과]
# 걸린시간 :  1657159951.857787
# loss :  [2944.085205078125, 0.0]
# r2스코어 :  0.505048018551157
