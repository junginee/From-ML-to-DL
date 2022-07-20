from datetime import datetime
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM, Input, Dropout
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime

import time
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt


# 1-1. 데이터 불러오기
path = "./_data/test_amore_0718/"
samsung = pd.read_csv(path + "삼성전자220718.csv", thousands=',', encoding='euc-kr')
amore = pd.read_csv(path + "아모레220718.csv", thousands=',', encoding='euc-kr')

#1-2. 데이터 확인
print(samsung.head())
print("=============================================================================================")
print(amore.head())

# 1-3. 데이터 오름차순 정렬
samsung = samsung.loc[::-1].reset_index(drop = True)
amore = amore.loc[::-1].reset_index(drop=True)

# 1-4. string 형태의 일자 index를 datetime으로 변경
samsung['일자'] = pd.to_datetime(samsung['일자'])
amore['일자'] = pd.to_datetime(amore['일자'])

print(samsung.info())
print("=============================================================================================")
print(amore.info())

# 1-5. '일자'를 연,월,일로 나누기 위한 연,월, 일 컬럼 추가
samsung.insert(0,'연',samsung['일자'].dt.year)
samsung.insert(1,'월',samsung['일자'].dt.month)
samsung.insert(2,'일',samsung['일자'].dt.day)

amore.insert(0,'연',amore['일자'].dt.year)
amore.insert(1,'월',amore['일자'].dt.month)
amore.insert(2,'일',amore['일자'].dt.day)

print(samsung.head())
print(amore.head())

# 1-5. '일자'를 연,월,일로 나누기 위한 연,월, 일 컬럼 추가
samsung.drop(columns='일자', axis=1, inplace = True, errors='ignore')
amore.drop(columns='일자', axis=1, inplace = True, errors='ignore')

# 1-6. 기존 '일자' 컬럼 제거
print(samsung.head())
print(amore.head())
print(amore.head())

# 1-7. 2022년 4월 이전 데이터 삭제
#1. 2022년 이전 데이터 삭제
delete_samsung_past_years = samsung[samsung['연']<2022].index
samsung.drop(delete_samsung_past_years, inplace = True, errors='ignore')
delete_amore_past_years = amore[amore['연']<2022].index
amore.drop(delete_amore_past_years,inplace = True, errors='ignore')

#2. 2022년 4월 이전 데이터 삭제
delete_samsung_past_months = samsung[samsung['월']<4].index  
samsung.drop(delete_samsung_past_months,inplace = True, errors='ignore')
delete_amore_past_months = amore[amore['월']<4].index
amore.drop(delete_amore_past_months, inplace = True, errors='ignore')


# 1-8. 주가 예측 시 필요해 보이지 않는 컬럼 삭제 및 사용할 컬럼 정의
# [필요 없는 칼럼 삭제] 전일비, Unnamed: 6, 등락률, 금액(백만), 신용비, 개인, 기관, 외인(수량), 외국계, 프로그램, 외인비
samsung.drop(columns=['전일비', 'Unnamed: 6', '등락률', '금액(백만)','신용비', '개인', '기관', '외인(수량)',  '외국계', '프로그램', '외인비'], axis=1, inplace=True, errors='ignore')
amore.drop( columns=['전일비', 'Unnamed: 6', '등락률', '금액(백만)','신용비', '개인', '기관', '외인(수량)',  '외국계', '프로그램', '외인비'], axis=1, inplace=True, errors='ignore')

print(samsung.head())
print("=============================================================================================")
print(amore.head())

# [사용할 컬럼 정의] 사용할 column을 정의(쓸 컬럼만 지정 )
samsung = samsung.loc[:, ['연','월','일','시가', '고가', '저가', '거래량','종가']]
amore = amore.loc[:, ['연','월','일','시가', '고가', '저가', '거래량','종가']]

# 1-9. 데이터 자르기
samsung = np.array(samsung)
amore = np.array(amore)


def split_xy(dataset, time_steps, y_column):
    x, y = list(), list()
    for i in range(len(dataset)):
        x_end_number = i + time_steps
        y_end_number = x_end_number + y_column-1
        
        if y_end_number > len(dataset):
            break
        tmp_x = dataset[i:x_end_number, 1:]
        tmp_y = dataset[x_end_number -1:y_end_number, -1]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x1, y1 = split_xy(samsung, 3, 3)
x2, y2 = split_xy(amore, 3, 3)

print(x1.shape, y1.shape)  
print(x2.shape, y2.shape) 

# 1-10. 데이터 reshape
x1_train, x1_test, x2_train, x2_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    x1, x2, y1, y2, train_size=0.7, shuffle=True, random_state=66)

x1_train = x1_train.reshape(x1_train.shape[0], x1_train.shape[1] * x1_train.shape[2])
x1_test  = x1_test.reshape(x1_test.shape[0],x1_test.shape[1]*x1_test.shape[2])

x2_train = x2_train.reshape(x2_train.shape[0], x2_train.shape[1] * x2_train.shape[2])
x2_test  = x2_test.reshape(x2_test.shape[0],x2_test.shape[1] * x2_test.shape[2])

print(x1_train.shape) 
print(x1_test.shape) 
print(x2_train.shape)
print(x2_test.shape)

# 1-11. 데이터 전처리
from sklearn.preprocessing import StandardScaler
data1 = StandardScaler()
data1.fit(x1_train)
x1_train_scale = data1.transform(x1_train)
x1_test_scale = data1.transform(x1_test)

data2 = StandardScaler()
data2.fit(x2_train)
x2_train_scale = data1.transform(x2_train)
x2_test_scale = data1.transform(x2_test)

# 1-12. 데이터 전처리 후 reshape
x1_train_scale = np.reshape(x1_train_scale,(x1_train_scale.shape[0], 3 , 7))
x1_test_scale = np.reshape(x1_test_scale,(x1_test_scale.shape[0], 3 , 7))
x2_train_scale = np.reshape(x2_train_scale,(x2_train_scale.shape[0], 3 , 7))
x2_test_scale = np.reshape(x2_test_scale,(x2_test_scale.shape[0], 3 , 7))


# 2-1. 모델구성(1)
input1 = Input((3,7))
dense1 = LSTM(10, activation='relu', name = 'dense1')(input1)
dense2 = Dense(22, activation='relu', name = 'dense2')(dense1)
dense3 = Dense(16, activation='relu', name = 'dense3')(dense2)
dense4 = Dense(20, activation='relu', name = 'dense4')(dense3)
output1 = Dense(16, name = 'output1')(dense4)

# 2-2. 2번 모델구성(2)
input2 = Input((3,7))
dense11 = LSTM(10, activation='relu', name = 'dense11')(input2)
dense12 = Dense(32, activation='relu', name = 'dense12')(dense11)
dense13 = Dropout(0.3, name = 'dense13')(dense11)
dense14 = Dense(14, activation='relu', name = 'dense14')(dense13)
dense15 = Dense(20, activation='relu', name = 'dense15')(dense14)
output2 = Dense(32, name = 'output11')(dense15)

# 2-3. 모델 앙상블
from keras.layers.merge import concatenate

merge1 = concatenate([output1, output2])
merge2 = Dense(130, activation='relu', name='mg2')(merge1)
output3 = Dense(1)(merge2)

# 2-4. 모델정의 및 summary 확인
model = Model(inputs=[input1, input2], outputs=output3)

model.summary()

# 3. 훈련된 모델 불러오기

model = load_model("./_test/amore/k46(5).hdf5")


# 4. 평가, 예측
loss = model.evaluate ([x1_test_scale, x2_test_scale], y2_test)
print('loss :', loss)

predict =model.predict([x1_test_scale, x2_test_scale])

print('22-07-20 종가 :', predict[-1],'원') 
