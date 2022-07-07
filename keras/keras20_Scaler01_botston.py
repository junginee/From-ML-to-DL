

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import numpy as np
import time

#1. 데이터

datasets = load_boston()
x = datasets.data
y = datasets.target

# print(np.min(x)) #0.0
# print(np.max(x)) #711.0
# x = (x-np.min(x)) /(np.max(x)-np.min(x))
# print(x[:10])

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.7, random_state = 66)

###############스캘러 방법#####################################
#scaler = StandardScaler()
#scaler = MinMaxScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)
print(np.min(x_train)) #0.0 
print(np.max(x_train)) #1.0

#유의할점? 스케일링 할 때에는 test set에 대해서만 Scaler 해주면 된다는 것이다. 
#즉, train set에 대해서 fit()한 Scaler 객체를 이용해서 test set을 변황해주면 된다는 것이다.
#왜?학습데이터로 fit()이 적용된 스케일링 기준 정보를 그대로 test data에 적용해야하며
#그렇지 않고 test data로 다시 새로운 스케일링 기준 정보를 만들게 되면
#train data와 test data의 스케일링 기준 정보가 서로 달라지기 때문에 올바른 예측 결과를 도출하지 못할 수 있다.
################################################################


#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=13))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(70, activation='sigmoid'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam', metrics=['mae'])

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=300, mode='auto', verbose=1, 
                              restore_best_weights=True)        

start_time = time.time()
hist = model.fit(x_train, y_train, epochs=500, batch_size=5,
                 validation_split=0.2,
                 callbacks=[earlyStopping],
                 verbose=1)

end_time = time.time() 

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)



print("걸린시간 : ", end_time)
from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('loss : ' , loss)
print('r2스코어 : ', r2)


#[과제] --- 완료
#1. scaler 하기 전
# 걸린시간 :  1657087789.771446
# loss :  [62.2312126159668, 5.917354106903076]
# r2스코어 :  0.24675134213800853

#2. MinMaxScaler()
# 걸린시간 :  1657089129.778837
# loss :  [20.694347381591797, 2.8725569248199463]
# r2스코어 :  0.7495149622874973

#3. StandardScaler()
# 걸린시간 :  1657089274.0064988
# loss :  [12.229043960571289, 2.4662506580352783]
# r2스코어 :  0.8519792672612871

#4. MaxAbsScaler()
# loss:  
# r2스코어 : 

#5. RobustScaler()
# loss: 
# r2스코어 : 







