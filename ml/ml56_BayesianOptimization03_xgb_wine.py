import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from bayes_opt import BayesianOptimization
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')
import time


#1. 데이터
datasets = load_wine()
x, y = datasets.data, datasets.target

# 라벨인코딩
# le = LabelEncoder()
# y = le.fit_transform(y)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=2022, train_size=0.8
)

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
              eval_metric='mlogloss',
              verbose=0,
              early_stopping_rounds=50
              )
    
    y_predict = model.predict(x_test)
    results = accuracy_score(y_test, y_predict)
    
    return results
lgb_bo = BayesianOptimization(f=lgb_hamsu, 
                              pbounds=bayesian_params,
                              random_state=1234,
                              )
lgb_bo.maximize(init_points=5,
                n_iter=20)
print(lgb_bo.max)
# {'target': 1.0, 'params': {'colsample_bylevel': 0.6714419750926706, 
# 'colsample_bynode': 0.49343775408860785, 'colsample_bytree': 0.0, 
# 'gamma': 12.203504373723936, 'max_depth': 9.985404526420755, 
# 'min_child_weight': 3.660893896504168, 'reg_alpha': 8.637529187820402, 
# 'reg_lambda': 0.0, 'subsample': 0.8157600553209446}}
'''

############################### 수정한 파라미터 이용 ##################################

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
              eval_metric='mlogloss',
              verbose=0,
              early_stopping_rounds=50
              )
    
    y_predict = model.predict(x_test)
    results = accuracy_score(y_test, y_predict)
    
    return results

lgb_bo = BayesianOptimization(f=lgb_hamsu, 
                              pbounds=bayesian_params,
                              random_state=2022,
                              )
lgb_bo.maximize(init_points=5,
                n_iter=20)
print(lgb_bo.max)

# {'target': 0.9444444444444444, 
#  'params': {'colsample_bylevel': 0.0, 'colsample_bynode': 1.0, 'colsample_bytree': 0.0, 
#             'gamma': 0.0, 'max_depth': 10.0, 'min_child_weight': 2.5209331437824356, 
#             'reg_alpha': 10.0, 'reg_lambda': 6.390046806785133, 'subsample': 0.9774959134870883}}
