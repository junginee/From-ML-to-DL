import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,KFold, cross_val_score ,StratifiedKFold
from sklearn.metrics import r2_score
from tqdm import tqdm_notebook
import warnings
warnings.filterwarnings("ignore")


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
print(train_set.isnull().sum()) 
train_set = train_set.fillna(train_set.mean()) 
print(train_set.isnull().sum())
print(train_set.shape)
 

test_set = test_set.fillna(test_set.mean())


x = train_set.drop(['SalePrice'], axis=1) 
print(x)
print(x.columns)
print(x.shape) # (1460, 79)

y = train_set['SalePrice']
print(y)
print(y.shape) # (1460, )

x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.75, shuffle=True, random_state=68)

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)
test_set = scaler.transform(test_set)
print(np.min(x_train))  # 0.0
print(np.max(x_train))  # 1.0
print(np.min(x_test))  # 1.0
print(np.max(x_test))  # 1.0

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle = True, random_state=66)


#2. 모델구성
model = LinearSVR() 
                

#3,4. 컴파일, 훈련, 평가, 예측

# model.fit(x_train, y_train)
scores = cross_val_score(model,x,y,cv=kfold)
print('ACC : ',scores,'\ncross_val_score :', round(np.mean(scores),4))

# ACC :  [0.71640948 0.50009411 0.77078097 0.66343296 0.76367513] 
# cross_val_score : 0.6829
