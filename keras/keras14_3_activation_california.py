# sklearn.datasets.fetch_california_housing

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

datasets = fetch_california_housing() 
x = datasets.data 
y = datasets.target 
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, shuffle=True, random_state=50)

# print(x)
# print(y)
# print(x.shape, y.shape) #(20640, 8) (20640,)

# print(datasets.feature_names)
# print(datasets.DESCR)

#2. 모델구성
model = Sequential()
model.add(Dense(5,input_dim=8))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(20))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(5, activation='relu'))
model.add(Dense(10, activation='sigmoid'))
model.add(Dense(1))
                

#3. 컴파일, 훈련

model.compile(loss='mse', optimizer='adam', metrics=['mae'])
from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=300, mode='auto', verbose=1, 
                              restore_best_weights=True)        

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=500, batch_size=50,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)

end_time = time.time()

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ',loss)

y_predict = model.predict(x_test) 
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict) #R 제곱 = 예측값 (y_predict) / 실제값 (y_test) 
print('r2스코어 : ', r2)


# 1. validation 적용했을때 결과
#loss : 0.6172688007354736
#r2스코어 : 0.5358171420467743

# 2. validation / EarlyStopping 적용했을때 결과
# loss :       
# r2 스코어 : 

# 3. validation / EarlyStopping / activation 적용했을때 결과
# loss : 
# r2 스코어 : 

# 1 > 2 > 3 과정을 거칠 수록 r2 값이 1에 가까워지는 것을 확인할 수 있다. 
# 이는 점차 모델 성능이 좋아졌음으로 해석할 수 있다.  
