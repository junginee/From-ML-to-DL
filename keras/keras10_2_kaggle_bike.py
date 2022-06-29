# https://www.kaggle.com/competitions/bike-sharing-demand
# 캐글 바이크 문제풀이
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#1. 데이터
path = './_data/kaggle_bike/' # 경로 = .현재폴더 /하단
train_set = pd.read_csv(path + 'train.csv', # train.csv 의 데이터가 train set에 들어가게 됨
                        index_col=0) # 0번째 컬럼은 인덱스로 지정하는 명령
print(train_set)
print(train_set.shape) # (10886, 11)

test_set = pd.read_csv(path + 'test.csv',
                       index_col=0)
sampleSubmission = pd.read_csv(path + 'sampleSubmission.csv',
                       index_col=0)
print(test_set)
print(test_set.shape) # (6493, 8) # 예측 과정에서 쓰일 예정

print(train_set.columns)
print(train_set.info()) # 각 컬럼에 대한 디테일한 내용 출력 / null값(중간에 빠진 값) '결측치'
print(train_set.describe())

#### 결측치 처리 1. 제거 ####
print(train_set.isnull().sum()) # 각 컬럼당 null의 갯수 확인가능
train_set = train_set.fillna(train_set.mean())
print(train_set.isnull().sum())
print(train_set.shape) # (10886, 11)


test_set = test_set.fillna(test_set.mean())


x = train_set.drop(['casual', 'registered', 'count'], axis=1) # axis는 'count'가 컬럼이라는 것을 명시하기 위해
print(x)
print(x.columns)
print(x.shape) # (10886, 8)

y = train_set['count']
print(y)
print(y.shape) # (10886, )

x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.99, shuffle=True, random_state=68)

#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=8))
model.add(Dense(100, activation='selu'))
model.add(Dense(100, activation='selu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam')
model.fit(x_train, y_train, epochs=330, batch_size=100)

#4. 평가 예측
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict) : #(원y값, 예측y값)
    return np.sqrt(mean_squared_error(y_test, y_predict))

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)


y_summit = model.predict(test_set)

print(y_summit)
print(y_summit.shape)


sampleSubmission['count'] = y_summit
sampleSubmission = abs(sampleSubmission)
sampleSubmission = sampleSubmission.fillna(sampleSubmission.mean())
sampleSubmission.to_csv(path + 'sampleSubmission_test04.csv', index=True)

# loss : 92.75188446044922
# RMSE :  139.62974866131475 