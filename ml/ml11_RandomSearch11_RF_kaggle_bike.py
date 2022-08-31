import numpy as np
import pandas as pd
from sqlalchemy import true
from sklearn.model_selection import train_test_split, KFold, GridSearchCV,cross_val_score,StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVR
from sklearn.linear_model import Perceptron, LinearRegression
import datetime as dt
import warnings
warnings.filterwarnings("ignore")

#1. 데이터
path = './_data/kaggle_bike/'
train_set = pd.read_csv(path + 'train.csv')            
test_set = pd.read_csv(path + 'test.csv')

######## Year, month, day, time separation ############

train_set["hour"] = [t.hour for t in pd.DatetimeIndex(train_set.datetime)]
train_set["day"] = [t.dayofweek for t in pd.DatetimeIndex(train_set.datetime)]
train_set["month"] = [t.month for t in pd.DatetimeIndex(train_set.datetime)]
train_set['year'] = [t.year for t in pd.DatetimeIndex(train_set.datetime)]
train_set['year'] = train_set['year'].map({2011:0, 2012:1})

test_set["hour"] = [t.hour for t in pd.DatetimeIndex(test_set.datetime)]
test_set["day"] = [t.dayofweek for t in pd.DatetimeIndex(test_set.datetime)]
test_set["month"] = [t.month for t in pd.DatetimeIndex(test_set.datetime)]
test_set['year'] = [t.year for t in pd.DatetimeIndex(test_set.datetime)]
test_set['year'] = test_set['year'].map({2011:0, 2012:1})

train_set.drop('datetime',axis=1,inplace=True)
train_set.drop('casual',axis=1,inplace=True) 
train_set.drop('registered',axis=1,inplace=True)
test_set.drop('datetime',axis=1,inplace=True)

x = train_set.drop(['count'], axis=1)  
print(x.shape) # (10886, 12)

y = train_set['count'] 
print(y.shape) # (10886,)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.75,
                                                    random_state=20
                                                    )

scaler = MinMaxScaler()
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
                   
model = GridSearchCV(RandomForestRegressor(), parameters, cv=kfold, verbose=1,          
                     refit=True, n_jobs=1)                                        


#3. 컴파일, 훈련
model.fit(x_train, y_train)

print("최적의 매개변수 : ", model.best_estimator_)
# 최적의 매개변수 :  RandomForestRegressor(min_samples_leaf=3, min_samples_split=5, n_estimators=200)
print("최적의 파라미터 : ", model.best_params_)
# 최적의 파라미터 :  {'min_samples_leaf': 3, 'min_samples_split': 5, 'n_estimators': 200}
print("best_score_ : ", model.best_score_)
# best_score_ :  0.9415037691847973
print("model.score : ", model.score(x_test, y_test))
# model.score :  0.9518660438883282

#4. 평가
y_predict = model.predict(x_test)
print("r2_score", round(r2_score(y_test, y_predict),4))
'''
# r2_score 0.9519
# y_pred_best = model.best_estimator_.__prepare__(x_test)
# print('최적 튠 ACC : ', accuracy_score(y_test, y_pred_best))
'''
