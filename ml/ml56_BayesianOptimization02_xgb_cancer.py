import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from bayes_opt import BayesianOptimization
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer, LabelEncoder
from sklearn.metrics import accuracy_score, r2_score
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import time


#1. 데이터
datasets = load_breast_cancer()
x, y = datasets.data, datasets.target
print(x.shape, y.shape)     # 

x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=123, train_size=0.8
)

# 라벨인코딩
le = LabelEncoder()
y = le.fit_transform(y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
'''
bayesian_params = {
    'max_depth' : [2,10], #default 3/ 0~inf(무한대) / 정수 => 소수점은 정수로 변환하여 적용해야 함
    'gamma': [0,100], #default 0 / 0~inf
    'min_child_weight': [0,100], #default 1 / 0~inf
    'subsample' : [0,1], #default 1 / 0~1
    'colsample_bytree' : [0,1], #default 1 / 0~1
    'colsample_bylevel' : [0,1], #default 1 / 0~1
    'colsample_bynode' : [0,1], #default 1 / 0~1
    'reg_alpha' : [0,10], #default 0 / 0~inf / L1 절대값 가중치 규제 / 그냥 alpha도 적용됨
    'reg_lambda' : [0,10], #default 1 / 0~inf / L2 제곱 가중치 규제 / 그냥 lambda도 적용됨
}
def lgb_hamsu(max_depth, gamma, min_child_weight,  
              subsample, colsample_bytree, colsample_bylevel, colsample_bynode, 
              reg_lambda, reg_alpha):
    params = {
        'n_estimators' : 500, 'learning_rate' : 0.02,
        'max_depth' : int(round(max_depth)),                # 무조건 정수
        'gamma' : int(round(gamma)), 
        'min_child_weight' : int(round(min_child_weight)),  
        'subsample' : max(min(subsample, 1), 0),             # 0~1 사이의 값
        'colsample_bytree' : max(min(colsample_bytree, 1), 0),   
        'colsample_bylevel' : max(min(colsample_bylevel, 1), 0),   
        'colsample_bynode' : max(min(colsample_bynode, 1), 0),   
        'reg_lambda' : max(reg_lambda, 0),          # 무조건 양수만
        'reg_alpha' : max(reg_alpha, 0),        
    }
    
    # *여러개의인자를받겠다
    # **키워드를 받겠다(딕셔너리 형태)
    model = XGBClassifier(**params)
    
    model.fit(x_train, y_train,
              eval_set=[(x_train, y_train), (x_test, y_test)],
              eval_metric='logloss',
              verbose=0,
              early_stopping_rounds=50
              )
    
    y_predict = model.predict(x_test)
    results = accuracy_score(y_test, y_predict)
    
    return results
lgb_bo = BayesianOptimization(f=lgb_hamsu, 
                              pbounds=bayesian_params,
                              random_state=123,
                              )
lgb_bo.maximize(init_points=5,
                n_iter=20)
print(lgb_bo.max)
# {'target': 0.9912280701754386, 'params': {'colsample_bylevel': 0.8085494320645987, 
# 'colsample_bynode': 0.9364960151161995, 'colsample_bytree': 0.36890313585418444, 
# 'gamma': 2.2562050580019255, 'max_depth': 4.0702880086299125, 
# 'min_child_weight': 2.1481232639242953, 'reg_alpha': 9.957165696511261, 
# 'reg_lambda': 2.0764054686027578, 'subsample': 0.9866844291313641}}
'''
############################### 수정한 파라미터 이용 ##################################

bayesian_params = {
    'max_depth' : [2,10], #default 3/ 0~inf(무한대) / 정수 => 소수점은 정수로 변환하여 적용해야 함
    'gamma': [0,100], #default 0 / 0~inf
    'min_child_weight': [0,100], #default 1 / 0~inf
    'subsample' : [0.9,1], #default 1 / 0~1
    'colsample_bytree' : [0,1], #default 1 / 0~1
    'colsample_bylevel' : [0.8,1], #default 1 / 0~1
    'colsample_bynode' : [0.9,1], #default 1 / 0~1
    'reg_alpha' : [9,100], #default 0 / 0~inf / L1 절대값 가중치 규제 / 그냥 alpha도 적용됨
    'reg_lambda' : [2,100], #default 1 / 0~inf / L2 제곱 가중치 규제 / 그냥 lambda도 적용됨
}

def lgb_hamsu(max_depth, gamma, min_child_weight,  
              subsample, colsample_bytree, colsample_bylevel, colsample_bynode, 
              reg_lambda, reg_alpha):
    params = {
        'n_estimators' : 500, 'learning_rate' : 0.02,
        'max_depth' : int(round(max_depth)),                # 무조건 정수
        'gamma' : int(round(gamma)), 
        'min_child_weight' : int(round(min_child_weight)),  
        'subsample' : max(min(subsample, 1), 0),             # 0~1 사이의 값
        'colsample_bytree' : max(min(colsample_bytree, 1), 0),   
        'colsample_bylevel' : max(min(colsample_bylevel, 1), 0),   
        'colsample_bynode' : max(min(colsample_bynode, 1), 0),   
        'reg_lambda' : max(reg_lambda, 0),          # 무조건 양수만
        'reg_alpha' : max(reg_alpha, 0),        
    }
    
    # *여러개의인자를받겠다
    # **키워드를 받겠다(딕셔너리 형태)
    model = XGBClassifier(**params)
    
    model.fit(x_train, y_train,
              eval_set=[(x_train, y_train), (x_test, y_test)],
              eval_metric='logloss',
              verbose=0,
              early_stopping_rounds=50
              )
    
    y_predict = model.predict(x_test)
    results = accuracy_score(y_test, y_predict)
    
    return results

lgb_bo = BayesianOptimization(f=lgb_hamsu, 
                              pbounds=bayesian_params,
                              random_state=123,
                              )        
lgb_bo.maximize(init_points=5,
                n_iter=20)
print(lgb_bo.max)

#{'target': 0.9912280701754386, 'params': {'colsample_bylevel': 1.0, 'colsample_bynode': 0.9, 
# 'colsample_bytree': 1.0, 'gamma': 0.0, 'max_depth': 10.0, # 'min_child_weight': 0.0, 
# 'reg_alpha': 20.07740204674745, 'reg_lambda': 2.0, 'subsample': 1.0}}
