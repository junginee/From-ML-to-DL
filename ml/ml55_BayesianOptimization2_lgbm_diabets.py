from dataclasses import dataclass
from bayes_opt import BayesianOptimization
from lightgbm import LGBMRegressor
import numpy as np

from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

import warnings
warnings.filterwarnings("ignore")


#1. 데이터
datasets = load_diabetes()
x, y = datasets.data, datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x,y, random_state=1234, train_size=0.8
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
bayesian_parames = {
              'max_depth' : (6,16),
              'num_leaves' : (24,64),
              'min_child_samples' :(10,200),
              'min_child_weight' :(1,50),
              'subsample' : (0.5,1),
              'colsample_bytree' : (0.5,1),
              'max_bin' : (10,500),
              'reg_alpha' : (0.01,50),
              'reg_lamda' : (0.001,10)
              }

def lgb_hamsu(max_depth, num_leaves, min_child_samples, min_child_weight,
              subsample, colsample_bytree, max_bin, reg_lamda, reg_alpha):
    params = {
              'n_estimators' : 500,'learning_rate' :0.02,
              'max_depth' : int(round(max_depth)),
              'num_leaves' : int(round(num_leaves)),
              'min_child_samples' :int(round(min_child_samples)),
              'min_child_weight' :int(round(min_child_weight)),
              'subsample' : max(min(subsample,1),0),        #0~1 사이의 값이 들어옴
              'colsample_bytree' : max(min(colsample_bytree,1),0),
              'max_bin' : max(int(round(max_bin)),10),  #무조건 10 이상
              'reg_alpha' : max(reg_alpha,0),           #무조건 양수민    
              'reg_lamda' : max(reg_lamda,0)
              }
    
    # *여러개의인자를받겠다 
    # **키워드받겠다(딕셔너리형태)
    model = LGBMRegressor(**params)
    
    model.fit(x_train, y_train,
              eval_set=[(x_train, y_train), (x_test, y_test)],
              eval_metric=0,
              verbose=0,
              early_stopping_rounds=50)
    y_predict = model.predict(x_test)
    results = r2_score(y_test, y_predict)
    
    return results

lgb_bo = BayesianOptimization(f=lgb_hamsu,
                              pbounds= bayesian_parames,
                              random_state= 1234)
lgb_bo.maximize(init_points=5, n_iter=20)

print(lgb_bo.max)

###############################################
#1. 수정한 파라미터로 모델 만들어서 비교
#2. 수정한 파라미터를 이용해서 파라미터 재조정