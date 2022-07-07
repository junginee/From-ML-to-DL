


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
import time

#1. 데이터
path = './_data/kaggle_house/' # 경로 = .현재폴더 /하단
train_set = pd.read_csv(path + 'train.csv', # train.csv 의 데이터가 train set에 들어가게 됨
                        index_col=0) # 0번째 컬럼은 인덱스로 지정하는 명령
#print(train_set)
#print(train_set.shape) # (1460, 80) 원래 열이 81개지만, id를 인덱스로 제외하여 80개

test_set = pd.read_csv(path + 'test.csv',
                       index_col=0)

drop_cols = ['Alley', 'PoolQC', 'Fence', 'MiscFeature'] # Columns with more than 70% of missing values
test_set.drop(drop_cols, axis = 1, inplace =True)

sample_submission = pd.read_csv(path + 'sample_submission.csv',
                       index_col=0)
#print(test_set)
#print(test_set.shape) # (1459, 79) # 예측 과정에서 쓰일 예정

train_set.drop(drop_cols, axis = 1, inplace =True)
cols = ['MSZoning', 'Street','LandContour','Neighborhood','Condition1','Condition2',
                'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Foundation',
                'Heating','GarageType','SaleType','SaleCondition','ExterQual','ExterCond','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',
                'BsmtFinType2','HeatingQC','CentralAir','Electrical','KitchenQual','Functional',
                'FireplaceQu','GarageFinish','GarageQual','GarageCond','PavedDrive','LotShape',
                'Utilities','LandSlope','BldgType','HouseStyle','LotConfig']

for col in tqdm_notebook(cols):
    le = LabelEncoder()
    train_set[col]=le.fit_transform(train_set[col])
    test_set[col]=le.fit_transform(test_set[col])

#print(train_set.columns)
#print(train_set.info()) # 각 컬럼에 대한 디테일한 내용 출력 / null값(중간에 빠진 값) '결측치'
#print(train_set.describe())

#### 결측치 처리 1. 제거 ####
print(train_set.isnull().sum()) # 각 컬럼당 null의 갯수 확인가능
train_set = train_set.fillna(train_set.mean()) # nan 값을 채우거나(fillna) 행별로 모두 삭제(dropna)
print(train_set.isnull().sum())
print(train_set.shape) # (1460, 80) 데이터가 얼마나 삭제된 것인지 확인가능(1460-1460=0)
 

test_set = test_set.fillna(test_set.mean())


x = train_set.drop(['SalePrice'], axis=1) # axis는 'count'가 컬럼이라는 것을 명시하기 위해
print(x)
print(x.columns)
print(x.shape) # (1460, 79)

y = train_set['SalePrice']
print(y)
print(y.shape) # (1460, )



x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.75, shuffle=True, random_state=68)

###############스캘러 방법#####################################
#scaler = StandardScaler()
#scaler = MinMaxScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)
test_set = scaler.transform(test_set)
print(np.min(x_train))  # 0.0
print(np.max(x_train))  # 1.0

print(np.min(x_test))  # 1.0
print(np.max(x_test))  # 1.0


#2. 모델구성
model = Sequential()
model.add(Dense(24, activation='linear', input_dim=75))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(10, activation='linear'))
model.add(Dense(50, activation='linear'))
model.add(Dense(1, activation='relu'))

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam',
              metrics=['accuracy'])
from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1, 
                              restore_best_weights=True) 
start_time = time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=100, 
                validation_split=0.2,
                callbacks=[earlyStopping],
                verbose=1)
end_time = time.time() 

#4. 평가 예측
print("걸린시간 : ", end_time)

loss = model.evaluate(x_test, y_test)
print('loss :', loss)

y_predict = model.predict(x_test)

def RMSE(y_test, y_predict) : #(원y값, 예측y값)
    return np.sqrt(mean_squared_error(y_test, y_predict)) # MSE에 루트를 씌워 돌려주겠다.

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)

y_summit = model.predict(test_set)

#[과제] --- train 파일 / test 파일 모두 transform!! 주의
#1. scaler 하기 전
# loss:  
# r2스코어 :

#2. MinMaxScaler()
# loss:  
# r2스코어 :

#3. StandardScaler()
# loss:  
# r2스코어 :

#4. MaxAbsScaler()
# loss:  
# r2스코어 : 

#5. RobustScaler()
# loss: 
# r2스코어 : 