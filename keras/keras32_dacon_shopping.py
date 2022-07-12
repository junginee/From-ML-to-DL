import numpy as np
import pandas as pd #read_csv, columns, info, describe, 결측치 제공
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

#1.데이터
path = './_data/shopping/'
train_set = pd.read_csv(path + 'train.csv') #id는 0번째에 위치한다. #[1459 rows x 10 columns]

print(train_set) #[6255 rows x 12 columns]
#print(train_set.shape) #(6255, 12)

test_set = pd.read_csv(path + 'test.csv') #예측에서 쓸것이다.
print(test_set) #[180 rows x 11 columns]
#print(test_set.shape)  #(180, 11)

# sample_submission 불러오기
sample_submission = pd.read_csv(path + 'sample_submission.csv')

train_set.tail()

#print(train_set.columns)
#print(train_set.info())  #결측치 : 데이터가 빠진 ..
#print(train_set.describe()) #[8 rows x 10 columns]

import matplotlib.pyplot as plt

# 이번엔 예측하고자 하는 값인 Weekly_Sales를 확인해봅니다.
plt.hist(train_set.Weekly_Sales, bins=50)
plt.show()

train_set = train_set.fillna(0)
#print(train_set)

# Date 칼럼에서 "월"에 해당하는 정보만 추출하여 숫자 형태로 반환하는 함수를 작성합니다.
def get_month(date):
    month = date[3:5]
    month = int(month)
    return month

# 이 함수를 Date 칼럼에 적용한 Month 칼럼을 만들어줍니다.
train_set['Month'] = train_set['Date'].apply(get_month)

# 결과를 확인합니다.
#print(train_set)

# IsHoliday 칼럼의 값을 숫자 형태로 반환하는 함수를 작성합니다.
def holiday_to_number(isholiday):
    if isholiday == True:
        number = 1
    else:
        number = 0
    return number

# 이 함수를 IsHoliday 칼럼에 적용한 NumberHoliday 칼럼을 만들어줍니다.
train_set['NumberHoliday'] = train_set['IsHoliday'].apply(holiday_to_number)

# 결과를 확인합니다.
#print(train_set)




# Date 전처리
test_set['Month'] = test_set['Date'].apply(get_month)

# IsHoliday 전처리
test_set['NumberHoliday'] = test_set['IsHoliday'].apply(holiday_to_number)

#from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# 결측치 처리

# 분석할 의미가 없는 칼럼을 제거합니다.
train_set = train_set.drop(columns=['id'])
test_set = test_set.drop(columns=['id'])

# 전처리 하기 전 칼럼들을 제거합니다.
train_set= train_set.drop(columns=['Date','IsHoliday'])
test_set = test_set.drop(columns=['Date','IsHoliday'])

# 학습에 사용할 정보와 예측하고자 하는 정보를 분리합니다.
x_train = train_set.drop(columns=['Weekly_Sales'])
y_train = train_set[['Weekly_Sales']]

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.7,
                                                    random_state=66
                                                    )


scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_set2 = scaler.transform(test_set2)

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
imputer.fit(train_set)
train_set_ = imputer.transform(train_set)

train_set = pd.DataFrame(train_set_, columns=train_set.columns)
print(train_set)


# 모델 선언
model = LinearRegression()

model.fit(x_train,y_train)

prediction = model.predict(test_set)
print('----------------------예측된 데이터의 상위 10개의 값 확인--------------------\n')
print(prediction[:10])

# 예측된 값을 정답파일과 병합
sample_submission ['Weekly_Sales'] = prediction

# 정답파일 데이터프레임 확인
sample_submission .head()

sample_submission .to_csv('submission.csv',index = False)
