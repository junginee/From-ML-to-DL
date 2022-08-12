from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor 
import time


#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, random_state=123, train_size=0.8, stratify=y
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle = True, random_state=123)


# 'n_estimators' : [100, 200, 300, 400, 500, 1000] #n_estimator = 디폴트 100 / 1 ~ inf   
# 'learning_rate' : [0.1, 0.2, 0.3, 0.5, 1, 0.01, 0.001]}  #디폴트 0.3 / 0~1
# 'max_dapth' : [None, 2, 3, 4, 5, 6, 7, 8, 9, 10]} 디폴트 6 / 0 ~ inf / 정수
# 'gamma' : [0, 1, 2, 3, 4, 5, 7, 10, 100]} 디폴트 0
# 'min_child_weight' :[0, 0.01, 0.001, 0.1, 0.5, 1, 5, 10, 100]} 디폴트 1
# 'subsample' : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1]} 디폴트 1 / 0~1
# 'colsample_bytree' : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1] 디폴트 1 / 0~1
# 'colsample_bylevel' : [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1]} 디폴트 1 / 0~1
# 'reg_alpha' : [0, 0.1, 0.01, 0.001, 1, 2, 10] / 디폴트 0 / 0~inf  / L1 절대값 가중치 규제 / alpha
# 'reg_lamda' : [0, 0.1, 0.01, 0.001, 1, 2, 10] / 디폴트 1 / 0~inf  / L2 제곱 가중치 규제 / lamda

'''
gamma 트리에서 가지를 추가로 치기 위해 필요한 최소한의 손실 감소 기준. 기준값이 클 수록 모형이 더 단순해진다.(> 0)
max_depth 트리의 최대 깊이.(> 0)
min_child_weight 트리에서 가지를 추가로 치기 위해 필요한 최소한의 사례 수.(> 0)
'''

parameters = {'n_estimators' : [100, 200, 300, 400, 500, 1000], 
              'learning_rate' : [0.1, 0.2, 0.3, 0.5, 1, 0.01, 0.001],
              'max_dapth' : [None, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              'gamma' : [0, 1, 2, 3, 4, 5, 7, 10, 100],
            #   'min_child_weight' :[0.5],
            #   'subsample' : [0.5],
            #   'colsample_bytree' : [0.3],
            #   'colsample_bylevel' : [1],
            #   'colsample_bynode' : [1],
              'reg_alpha' :[0, 0.1, 0.01, 0.001, 1, 2, 10],
              'reg_lamda' : [0, 0.1, 0.01, 0.001, 1, 2, 10]}


#max_depth 숫자가 작을수록 통상적으로 성능이 좋다. 숫자가 커질수록 과적합 걸릴 수 有                                                      
#min_depth 숫자가 커질수록 통상적으로 성능이 좋다. 숫자가 작아질수록 과적합 걸릴 수 有

#2. 모델
xgb = XGBClassifier(random_state = 123,
                    )

model = GridSearchCV(xgb, parameters, cv= kfold, n_jobs=8)

#3. 훈련
model.fit(x_train, y_train)

print("최상의 매개변수 :", model.best_params_)
print("최상의 점수 : ", model.best_score_)

#4. 평가
y_predict = model.predict(x_test)
print('r2 score :', round(r2_score(y_test, y_predict),4))

###############################################
# r2 score : 
###############################################


# 최상의 매개변수 : 
# 최상의 점수 :  

