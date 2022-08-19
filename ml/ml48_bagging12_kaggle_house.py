import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler, StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from tqdm import tqdm_notebook

#1. 데이터
path = './_data/kaggle_house/' 
train_set = pd.read_csv(path + 'train.csv', 
                        index_col=0) 


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


for col in tqdm_notebook(cols):
    le = LabelEncoder()
    train_set[col]=le.fit_transform(train_set[col])
    test_set[col]=le.fit_transform(test_set[col])


#### 결측치  제거 ####
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


x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=66, train_size=0.8, shuffle=True)

#2. 모델
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVC

scaler = StandardScaler()
# scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


model = LinearSVC(), LinearRegression(),DecisionTreeRegressor(),RandomForestRegressor()

for i in model :
    model = i
    new_model = BaggingRegressor(i,
                          n_estimators=100, 
                          n_jobs=-1,
                          random_state=123
                          )
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    print(i,'모델 score :',round(r2_score(y_test, y_predict),4),'\n') 
    
# LinearSVC() 모델 score : 0.2386 

# LinearRegression() 모델 score : 0.8223

# DecisionTreeRegressor() 모델 score : 0.7631 

# RandomForestRegressor() 모델 score : 0.883    