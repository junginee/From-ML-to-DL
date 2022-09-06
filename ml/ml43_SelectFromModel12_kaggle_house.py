import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
import time
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm_notebook
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score
from sklearn.feature_selection import SelectFromModel


#1. 데이터
path = './_data/kaggle_house/' 
train_set = pd.read_csv(path + 'train.csv',
                        index_col=0) 
#print(train_set)
#print(train_set.shape) # (1460, 80) 

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

#### 결측치 처리 ####
# print(train_set.isnull().sum()) 
train_set = train_set.fillna(train_set.mean()) 
# print(train_set.isnull().sum())
# print(train_set.shape) 
test_set = test_set.fillna(test_set.mean())

x = train_set.drop(['SalePrice'], axis=1) 
# print(x)
# print(x.columns)
# print(x.shape) # (1460, 79)

y = train_set['SalePrice']
# print(y)
# print(y.shape) # (1460, )

x_train, x_test, y_train, y_test = train_test_split(x, y,
    shuffle=True, random_state=123, train_size=0.8, #stratify=y
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# n_splits = 5
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=123)


#2. 모델
model = XGBRegressor(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0,
                      random_state=123, 
                      n_estimators=100,
                      learning_rate = 0.1,
                      max_depth = 5,
                      gamma = 0, 
                      min_child_weight = 0,
                      subsample = 1,
                      colsample_bytree = 1,
                      colsample_bylevel = 1,
                      colsample_bynode = 1,
                      reg_alpha = 1,
                      reg_lambda = 1,)

# model = GridSearchCV(xgb, parameters, cv=kfold, n_jobs=8)

model.fit(x_train, y_train, early_stopping_rounds=10, 
          eval_set=[(x_train, y_train), (x_test, y_test)],
          eval_metric=['logloss']
        )


results = model.score(x_test, y_test)
print('최종 점수 : ', results)

x_predict = model.predict(x_test)
r2 = r2_score(y_test, x_predict)
print('진짜 최종 점수 : ', r2)

print(model.feature_importances_)

thresholds = model.feature_importances_
print("=======================================")

##########
bscore = 0
idx_ = 0
##########

# for thresh in thresholds:
for i in range(len(thresholds)):
    selection = SelectFromModel(model, threshold=thresholds[i], prefit=True)
    
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    print(select_x_train.shape, select_x_test.shape) # (353, 9) (89, 9)
    
    selection_model = XGBRegressor(n_jobs=-1,
                                   random_state=123,
                                   n_estimators=100,
                                   learning_rate = 0.1,
                                   max_depth = 3,
                                   gamma = 1)
    
    selection_model.fit(select_x_train, y_train)
    
    y_predict = selection_model.predict(select_x_test)
    score = r2_score(y_test, y_predict)
    
    print("Thresh=%.3f, n=%d, R2: %.2f%%"
            %(thresholds[i], select_x_train.shape[1], score*100))

    if score >= bscore:
        bscore = score
        idx_= i

f_to_drop = []
for i in range(len(thresholds)):
    if thresholds[idx_]>=thresholds[i]:
        f_to_drop.append(i)
        
print(f_to_drop)
# [1, 3, 8]

x2_train = np.delete(x_train, f_to_drop, axis=1)
x2_test = np.delete(x_test, f_to_drop, axis=1)

model.fit(x2_train, y_train, early_stopping_rounds=10, 
          eval_set=[(x2_train, y_train), (x2_test, y_test)],
          )

print('드랍 후 테스트 스코어: ', model.score(x2_test, y_test))

score = r2_score(y_test, model.predict(x2_test))
print('드랍 후 r2_score 결과: ', score)


# 드랍 후 테스트 스코어:  0.8939583493278189
# 드랍 후 r2_score 결과:  0.8939583493278189
