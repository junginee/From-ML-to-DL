import numpy as np
import time
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.datasets import load_wine
from sklearn.preprocessing import OneHotEncoder
from sklearn.experimental import enable_halving_search_cv  
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, HalvingRandomSearchCV
from sklearn.metrics import accuracy_score
                        
#1. 데이터
datasets = load_wine()
x = datasets.data
y= datasets.target

print(x.shape, y.shape) #(178, 13) #(178,)
print(np.unique(y, return_counts = True)) #[0 1 2]

import tensorflow as tf
tf.random.set_seed(66)

x_train, x_test, y_train, y_test = train_test_split( x, y, train_size = 0.8, shuffle=True, random_state=120 )

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)
print(np.min(x_train)) #0.0 
print(np.max(x_train)) #1.0

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle = True, random_state=66)  

parameters = [
        {'n_estimators':[100,200], 'max_depth':[6,8,10,12], 'min_samples_split':[2,3,5,10]},
        {'n_estimators':[100,200], 'min_samples_leaf':[3,5,7,10], 'min_samples_split':[2,3,5,10]},
]                                                                     


#2. 모델구성
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron, LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# model =  RandomForestClassifier(max_depth=8, min_samples_split=3, n_estimators=200)                     
model =HalvingRandomSearchCV(RandomForestClassifier(), parameters, cv=kfold, verbose=1,          
                     refit=True, n_jobs=1)                            
                                                                           
                                                                          
#3. 컴파일, 훈련
model.fit(x_train, y_train)

####### GridSearchCV 탐색결과 #######
# print("최적의 매개변수 : ", model.best_estimator_)
# 최적의 매개변수 :  RandomForestClassifier(max_depth=8, min_samples_split=3, n_estimators=200)
# print("최적의 파라미터 : ", model.best_params_)
# 최적의 파라미터 :  {'max_depth': 8, 'min_samples_split': 3, 'n_estimators': 200}
# print("best_score_ : ", model.best_score_)
# best_score_ :  0.9790640394088671
# print("model.score : ", model.score(x_test, y_test))
# model.score :  1.0
###################################### 

### HalvingRandomSearchCV 탐색결과 ###
print("최적의 매개변수 : ", model.best_estimator_)
# 최적의 매개변수 :  RandomForestClassifier(max_depth=6)
print("최적의 파라미터 : ", model.best_params_)
# 최적의 파라미터 :  {'n_estimators': 100, 'min_samples_split': 2, 'max_depth': 6}
print("best_score_ : ", model.best_score_)
# best_score_ :  0.977124183006536
print("model.score : ", model.score(x_test, y_test))
# model.score :  1.0

###################################### 

#4. 평가
y_predict = model.predict(x_test)
print("accuracy_score", round(accuracy_score(y_test, y_predict),4))
# accuracy_score 1.0

# y_pred_best = model.best_estimator_.__prepare__(x_test)
# print('최적 튠 ACC : ', accuracy_score(y_test, y_pred_best))
