import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import ridSearchCV, StratifiedKFold,RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv   
from sklearn.model_selection import HalvingRandomSearchCV
import warnings
warnings.filterwarnings('ignore')

#1. 데이터

path = './_data/kaggle_bike/'
train_set = pd.read_csv(path + 'train.csv')
test_set = pd.read_csv(path + 'test.csv')
train_set['datetime'] = pd.to_datetime(train_set['datetime']) 
train_set['year'] = train_set['datetime'].dt.year
train_set['month'] = train_set['datetime'].dt.month
train_set['day'] = train_set['datetime'].dt.day
train_set['hour'] = train_set['datetime'].dt.hour
train_set.drop(['datetime', 'day', 'year'], inplace=True, axis=1)
train_set['month'] = train_set['month'].astype('category')
train_set['hour'] = train_set['hour'].astype('category')
train_set = pd.get_dummies(train_set, columns=['season','weather'])
train_set.drop(['casual', 'registered'], inplace=True, axis=1)
train_set.drop('atemp', inplace=True, axis=1)
test_set['datetime'] = pd.to_datetime(test_set['datetime'])
test_set['month'] = test_set['datetime'].dt.month
test_set['hour'] = test_set['datetime'].dt.hour
test_set['month'] = test_set['month'].astype('category')
test_set['hour'] = test_set['hour'].astype('category')
test_set = pd.get_dummies(test_set, columns=['season','weather'])
drop_feature = ['datetime', 'atemp']
test_set.drop(drop_feature, inplace=True, axis=1)

x = train_set.drop(['count'], axis=1)
y = train_set['count']



x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.8, shuffle=True, random_state=666)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=66)

parameters = [
    {'n_estimators' : [100,200,300,400,500], 'max_depth' : [6,10,12,14,16]},                      
    {'max_depth' : [6, 8, 10, 12, 14], 'min_samples_leaf' : [3, 5, 7, 10, 12]},         
    {'min_samples_leaf' : [3, 5, 7, 10, 12], 'min_samples_split' : [2, 3, 5, 10, 12]},  
    {'min_samples_split' : [2, 3, 5, 10, 12]},                                     
    {'n_jobs' : [-1, 2, 4, 6],'min_samples_split' : [2, 3, 5, 10, 12]}             
]


    
#2. 모델구성
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# model = SVC(C=1, kernel='linear', degree=3)
model = HalvingRandomSearchCV(RandomForestRegressor(),parameters, cv=kfold, verbose=1,            
                     refit=True, n_jobs=-1)                            
                                                                    
#3. 컴파일, 훈련
import time
start = time.time()
model.fit(x_train, y_train)
end = time.time()

print("최적의 매개변수 : ", model.best_estimator_)  # 가장 좋은 추정치
# 최적의 매개변수 :  SVC(C=100, gamma=0.001)
print("최적의 파라미터 : ", model.best_params_)
# 최적의 파라미터 :  {'C': 100, 'gamma': 0.001, 'kernel': 'rbf'}

print("best_score_ : ", model.best_score_)        # 정확도
# best_score_ :  0.9666666666666668
print("model.score : ", model.score(x_test, y_test))
# model.score :  0.9666666666666667


#4. 평가, 예측
y_predict = model.predict(x_test)
print("r2_score : ", r2_score(y_test, y_predict))
# accuracy_score :  0.9666666666666667

y_pred_best = model.best_estimator_.predict(x_test)
print("최적 튠 R2 : ", r2_score(y_test,y_pred_best))
# 최적 튠 ACC :  0.9666666666666667
print("걸린시간 : ", round(end-start, 4))

# GridSearchCV
# 최적의 매개변수 :  RandomForestRegressor(min_samples_split=5, n_jobs=6)
# 최적의 파라미터 :  {'min_samples_split': 5, 'n_jobs': 6}
# best_score_ :  0.8514855020037586
# model.score :  0.8552395463089997
# r2_score :  0.8552395463089997
# 최적 튠 R2 :  0.8552395463089997
# 걸린시간 :  149.2676

# RandomizedSearchCV
# 최적의 매개변수 :  RandomForestRegressor(min_samples_split=3, n_jobs=4)
# 최적의 파라미터 :  {'n_jobs': 4, 'min_samples_split': 3}
# best_score_ :  0.8503210736464121
# model.score :  0.8561548624063268
# r2_score :  0.8561548624063268
# 최적 튠 R2 :  0.8561548624063268
# 걸린시간 :  12.0289

# HalvingGridSearchCV
# 최적의 매개변수 :  RandomForestRegressor(max_depth=16, n_estimators=500)
# 최적의 파라미터 :  {'max_depth': 16, 'n_estimators': 500}
# best_score_ :  0.8508313450563041
# model.score :  0.8562863519384702
# r2_score :  0.8562863519384702
# 최적 튠 R2 :  0.8562863519384702
# 걸린시간 :  69.1142

# HalvingRandomSearchCV
# 최적의 매개변수 :  RandomForestRegressor(max_depth=10, n_estimators=200)
# 최적의 파라미터 :  {'n_estimators': 200, 'max_depth': 10}
# best_score_ :  0.7554145443610379
# model.score :  0.8382699867281912
# r2_score :  0.8382699867281912
# 최적 튠 R2 :  0.8382699867281912
# 걸린시간 :  27.2018
