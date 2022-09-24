import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import time

datasets =load_diabetes()
x = datasets.data
y = datasets.target
x_train,x_test,y_train, y_test = train_test_split(x,y, train_size=0.75, shuffle=True, random_state=72)


scaler = StandardScaler()
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
model = Sequential()
model.add(Dense(6,input_dim=10))
model.add(Dense(15, activation='sigmoid'))
model.add(Dense(15, activation='relu'))
model.add(Dense(20))
model.add(Dense(20, activation='relu'))
model.add(Dense(20))
model.add(Dense(50, activation='sigmoid'))
model.add(Dense(1))

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

model.save_weights("./_save/keras23_9_save_diabets.h5")

#4.평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ',loss)

print("걸린시간 : ", end_time)
y_predict = model.predict(x_test) 

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict) #R 제곱 = 예측값 (y_predict) / 실제값 (y_test) 
print('r2스코어 : ', r2)

