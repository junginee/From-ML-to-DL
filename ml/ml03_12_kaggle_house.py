import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tqdm import tqdm_notebook


#1. 데이터
path = './_data/kaggle_house/' 
train_set = pd.read_csv(path + 'train.csv', index_col=0) 
test_set = pd.read_csv(path + 'test.csv', index_col=0)


drop_cols = ['Alley', 'PoolQC', 'Fence', 'MiscFeature'] 
test_set.drop(drop_cols, axis = 1, inplace =True)


sample_submission = pd.read_csv(path + 'sample_submission.csv', index_col=0)

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


#### Remove missing values ####
print(train_set.isnull().sum()) 
train_set = train_set.fillna(train_set.mean()) 
print(train_set.isnull().sum())
print(train_set.shape)
 

test_set = test_set.fillna(test_set.mean())


x = train_set.drop(['SalePrice'], axis=1) 
y = train_set['SalePrice']

x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.75, shuffle=True, random_state=68)

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)
test_set = scaler.transform(test_set)


#2. 모델구성
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LinearRegression  
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn. ensemble import RandomForestRegressor

model = Perceptron(),LinearSVC(),LinearRegression(),KNeighborsRegressor(),DecisionTreeRegressor(),RandomForestRegressor()


for i in model:    
    model = i
    

    #3. 컴파일, 훈련
    model.fit(x_train, y_train)


    #4. 평가, 예측
    result = model.score(x_test,y_test)   
    y_predict = model.predict(x_test)
    print(f"{i} : ", round(result,4))

# Perceptron() :  0.0027
# LinearSVC() :  0.0082
# LinearRegression() :  0.8373
# KNeighborsRegressor() :  0.6147
# DecisionTreeRegressor() :  0.7699
# RandomForestRegressor() :  0.8837
