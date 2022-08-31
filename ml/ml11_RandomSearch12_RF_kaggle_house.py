import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, train_test_split,GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score
from tqdm import tqdm_notebook
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score

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

n_splits=5
kfold = KFold(n_splits=5, shuffle=True, random_state=101)

parameters = [
        {'n_estimators':[100,200], 'max_depth':[6,8,10,12], 'min_samples_split':[2,3,5,10]},
        {'n_estimators':[100,200], 'min_samples_leaf':[3,5,7,10], 'min_samples_split':[2,3,5,10]},
]                                                                    

#2. 모델구성
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron, LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
                
model = RandomizedSearchCV(RandomForestRegressor(), parameters, cv=kfold, verbose=1,          
                     refit=True, n_jobs=1)                                       


#3. 컴파일, 훈련
model.fit(x_train, y_train)

print("최적의 매개변수 : ", model.best_estimator_)
# 최적의 매개변수 :  RandomForestRegressor(max_depth=12, min_samples_split=5, n_estimators=200)
print("최적의 파라미터 : ", model.best_params_)
# 최적의 파라미터 :  {'n_estimators': 200, 'min_samples_split': 5, 'max_depth': 12}
print("best_score_ : ", model.best_score_)
# best_score_ :  0.8320004092815483
print("model.score : ", model.score(x_test, y_test))
# model.score :  0.8881118991284918

#4. 평가
y_predict = model.predict(x_test)
print("r2_score", round(r2_score(y_test, y_predict),4))

# r2_score 0.8881
# y_pred_best = model.best_estimator_.__prepare__(x_test)
# print('최적 튠 ACC : ', accuracy_score(y_test, y_pred_best))
