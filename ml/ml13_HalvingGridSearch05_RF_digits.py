import numpy as np
import time
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn. datasets import load_digits
from sklearn.preprocessing import OneHotEncoder
from sklearn.experimental import enable_halving_search_cv  #얘가 더 위에 있어야 함
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, HalvingRandomSearchCV
from sklearn.metrics import accuracy_score
                        
#1. 데이터
datasets = load_digits()
x = datasets.data
y= datasets.target

print(x.shape, y.shape)
print(np.unique(y)) 

import tensorflow as tf
tf.random.set_seed(66)

x_train, x_test, y_train, y_test = train_test_split( x, y, train_size = 0.8, shuffle=True, random_state=68 )

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)
print(np.min(x_train)) 
print(np.max(x_train))

n_splits=5
kfold = KFold(n_splits=5, shuffle=True, random_state=101)

parameters = [
        {'n_estimators':[100,200], 'max_depth':[6,8,10,12]},
        {'n_estimators':[100,200], 'min_samples_leaf':[3,5,7,10], 'min_samples_split':[2,3,5,10]},
]                                                                       # 총 42번


#2. 모델구성
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron, LogisticRegression # LogisticRegression는 분류임
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# model =  RandomForestClassifier(min_samples_leaf=3, min_samples_split=5)                        
model = HalvingRandomSearchCV(RandomForestClassifier(), parameters, cv=kfold, verbose=1,          
                     refit=True, n_jobs=1)                             
                                                                           
                                                                          
                                                                           # 컴퓨터는 뜨거워지겠지만, 속도는 많이 빨라진다.


#3. 컴파일, 훈련
model.fit(x_train, y_train)

####### GridSearchCV 탐색결과 #######
# print("최적의 매개변수 : ", model.best_estimator_)
# 최적의 매개변수 :  RandomForestClassifier(min_samples_leaf=3, min_samples_split=5)
# print("최적의 파라미터 : ", model.best_params_)
# 최적의 파라미터 :  {'min_samples_leaf': 3, 'min_samples_split': 5, 'n_estimators': 100}
# print("best_score_ : ", model.best_score_)
# best_score_ :  0.97147212543554
# print("model.score : ", model.score(x_test, y_test))
# model.score :  0.9638888888888889
###################################### 

### HalvingRandomSearchCV 탐색결과 ###
print("최적의 매개변수 : ", model.best_estimator_)
# 최적의 매개변수 :  RandomForestClassifier(max_depth=10, n_estimators=200)
print("최적의 파라미터 : ", model.best_params_)
# 최적의 파라미터 :  {'n_estimators': 200, 'max_depth': 10}
print("best_score_ : ", model.best_score_)
# best_score_ :  0.9643202979515829
print("model.score : ", model.score(x_test, y_test))
# model.score :  0.9722222222222222

###################################### 

#4. 평가
y_predict = model.predict(x_test)
print("accuracy_score", round(accuracy_score(y_test, y_predict),4))
# accuracy_score 0.9722

# y_pred_best = model.best_estimator_.__prepare__(x_test)
# print('최적 튠 ACC : ', accuracy_score(y_test, y_pred_best))