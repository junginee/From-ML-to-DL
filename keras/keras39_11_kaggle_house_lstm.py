


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.models import Sequential, load_model, Model
from tensorflow.python.keras.layers import Dense,Dropout, LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


#1. 데이터
path = './_data/kaggle_house/' # 경로 = .현재폴더 /하단
train_set = pd.read_csv(path + 'train.csv', # train.csv 의 데이터가 train set에 들어가게 됨
                        index_col=0) 
#print(train_set)
#print(train_set.shape) # (1460, 80) 

test_set = pd.read_csv(path + 'test.csv',
                       index_col=0)

drop_cols = ['Alley', 'PoolQC', 'Fence', 'MiscFeature'] 
test_set.drop(drop_cols, axis = 1, inplace =True)

sample_submission = pd.read_csv(path + 'sample_submission.csv',
                       index_col=0)
#print(test_set)
#print(test_set.shape) # (1459, 79) 

train_set.drop(drop_cols, axis = 1, inplace =True)
cols = ['MSZoning', 'Street','LandContour','Neighborhood','Condition1','Condition2',
                'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','Foundation',
                'Heating','GarageType','SaleType','SaleCondition','ExterQual','ExterCond','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1',
                'BsmtFinType2','HeatingQC','CentralAir','Electrical','KitchenQual','Functional',
                'FireplaceQu','GarageFinish','GarageQual','GarageCond','PavedDrive','LotShape',
                'Utilities','LandSlope','BldgType','HouseStyle','LotConfig']

from tqdm import tqdm_notebook
for col in tqdm_notebook(cols):
    le = LabelEncoder()
    train_set[col]=le.fit_transform(train_set[col])
    test_set[col]=le.fit_transform(test_set[col])

#print(train_set.columns)
#print(train_set.info()) # 각 컬럼에 대한 디테일한 내용 출력 / null값(중간에 빠진 값) '결측치'
#print(train_set.describe())

#### 결측치 처리 1. 제거 ####
print(train_set.isnull().sum())
train_set = train_set.fillna(train_set.mean()) 
print(train_set.isnull().sum())
print(train_set.shape) # (1460, 80) 
 

test_set = test_set.fillna(test_set.mean())


x = train_set.drop(['SalePrice'], axis=1) # axis는 'count'가 컬럼이라는 것을 명시하기 위해
print(x)
print(x.columns)


y = train_set['SalePrice']
print(y)



x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.75, shuffle=True, random_state=68)
print(x_train.shape, x_test.shape) #(1095, 75) (365, 75)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)
test_set = scaler.transform(test_set)

x_train = x_train.reshape(1095,15,5)
x_test = x_test.reshape(365,15,5)
print(x_train.shape,x_test.shape)


#2. 모델구성
model = Sequential()
model.add(LSTM(64, return_sequences = True, input_shape=(15,5))) 
model.add(LSTM(7, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(7, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(32, activation='relu'))
model.add(Dense(1)) 

#3. 컴파일, 훈련
model.compile(loss='mae', optimizer='adam',
              metrics=['accuracy'])

from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1, 
                              restore_best_weights=True) 

hist = model.fit(x_train, y_train, epochs=1000, batch_size=100, 
                validation_split=0.2,
                callbacks=[earlyStopping],
                verbose=1)




#4. 평가 예측
loss = model.evaluate(x_test, y_test) 
print('loss : ', round(loss[0],4))

y_predict = model.predict(x_test)


def RMSE(a, b): 
    return np.sqrt(mean_squared_error(a, b))

rmse = RMSE(y_test, y_predict)
print("RMSE : ", round(rmse,4))

#================================= loss,RMSE =================================#
# loss :  56920.9219
# RMSE :  82365.1743
#=============================================================================#
