from bayes_opt import BayesianOptimization
from lightgbm import LGBMRegressor,LGBMClassifier
import numpy as np 
from xgboost import XGBClassifier,XGBRegressor

from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,r2_score
import warnings
warnings.filterwarnings('ignore')

#1.데이터 
datasets = load_digits()
x,y = datasets.data, datasets.target
x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8,random_state=123,shuffle=True,stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 'n_estimators' : [100, 200, 300, 400, 500, 1000] # 디폴트 100 / 1~inf  (inf: 무한대)
# 'learning_rate': [0.1, 0.2, 0.3, 0.5, 1, 0.01, 0.001] 디폴트 0.3/ 0~1 / eta라고 써도 먹힘
# 'max_depth': [None, 2, 3, 4, 5, 6, 7, 8, 9, 10] 디폴트 6 / 0~ inf / 정수
# 'gamma': [0, 1, 2, 3, 4, 5, 7, 10, 100] 디폴트 0/ 0~inf
# 'min_child_weight': [0, 0.01, 0.001, 0.1, 0.5, 1, 5, 10] 디폴트 1 / 0~inf
# 'subsample': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] 디폴트 1 / 0~1
# 'colsample_bytree': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] 디폴트 1 / 0~1
# 'colsample_bylevel': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] 디폴트 1 / 0~1
# 'colsample_bynode': [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] 디폴트 1 / 0~1
# 'reg_alpha': [0, 0.1, 0.01, 0.001, 1, 2, 10] 디폴트 0/ 0~inf / L1 절대값 가중치 규제 /alpha
# 'reg_lambda':[0, 0.1, 0.01, 0.001, 1, 2, 10] 디폴트 1/ 0~inf/ L2 제곱 가중치 규제 /lambda


bayesian_params ={'max_depth':(2,16),'gamma': (0,100),'min_child_weight':(1,50),'subsample':(0.1,1),
                  'colsample_bytree':(0.1,1),'colsample_bylevel':(0.1,1),'colsample_bynode':(0.1,1),'max_bin':(10,500),'reg_lambda':(0.001,10),'reg_alpha':(0.01,50)}


def lgb_hamsu(max_depth, gamma, min_child_weight,subsample,colsample_bytree,colsample_bylevel,colsample_bynode,max_bin, reg_lambda, reg_alpha):
    params ={'n_estimators':300, 'learning_rate':0.1,
             'max_depth':int(round(max_depth)),                  # 무조건 정수
             'gamma': int(round(gamma)),
             
             'min_child_weight': int(round(min_child_weight)),  
             'subsample': max(min(subsample,1),0),              # 어떤 값을 넣어도 0~1 의 값
             'colsample_bytree': max(min(colsample_bytree,1),0),
             'colsample_bylevel': max(min(colsample_bylevel,1),0),
             'colsample_bynode': max(min(colsample_bynode,1),0),
             'max_bin': max(int(round(max_bin)),10),            # 무조건 10이상
             'reg_lambda': max(reg_lambda,0),                   # 무조건 0이상(양수)
             'reg_alpha':max(reg_alpha,0)                       
    }

    #  * : 여러개의인자를받겠다
    # ** : 키워드받겠다(딕셔너리형태)
    model = XGBClassifier(**params) 

    model.fit(x_train,y_train,
              eval_set=[(x_train,y_train),(x_test,y_test)],
            #   eval_metric='merror',
              verbose=0,
              early_stopping_rounds=50)

    y_predict = model.predict(x_test)
    results = accuracy_score(y_test,y_predict)

    return results

lgb_bo = BayesianOptimization(f=lgb_hamsu,
                              pbounds=bayesian_params,
                              random_state=123)

lgb_bo.maximize(init_points=5, n_iter=100)  #초기 2번  ,n-iter : 20번 돌거다! 총 22번 돈다 

print(lgb_bo.max) 