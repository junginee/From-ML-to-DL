import numpy as np
import time
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn. datasets import fetch_covtype 
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
                        
#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y= datasets.target

print(datasets.feature_names)
print(datasets.DESCR)
print(x.shape, y.shape) #(581012, 54) (581012,)
print(np.unique(y, return_counts = True)) # y :[1 2 3 4 5 6 7]  / return_counts :[211840, 283301,  35754,   2747,   9493,  17367,  20510]

# import pandas as pd
# y = pd.get_dummies(y)
# print(y)

x_train, x_test, y_train, y_test = train_test_split( x, y, train_size = 0.8, shuffle=True, random_state=68 )


n_splits=5
kfold = KFold(n_splits=5, shuffle=True, random_state=101)

parameters = [
        {'n_estimators':[100,200], 'max_depth':[6,8,10,12], 'min_samples_split':[2,3,5,10]},
        {'n_estimators':[100,200], 'min_samples_leaf':[3,5,7,10], 'min_samples_split':[2,3,5,10]},
]                                                                       # 총 42번


#2. 모델구성
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron, LogisticRegression # LogisticRegression는 분류임
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

model =  RandomForestClassifier(max_depth=10, min_samples_split=3)                         
# model = GridSearchCV(RandomForestClassifier(), parameters, cv=kfold, verbose=1,          
#                      refit=True, n_jobs=1)                             
                                                                           
                                                                          
                                                                           # 컴퓨터는 뜨거워지겠지만, 속도는 많이 빨라진다.


#3. 컴파일, 훈련
model.fit(x_train, y_train)

####### GridSearchCV 탐색결과 #######
print("최적의 매개변수 : ", model.best_estimator_)

print("최적의 파라미터 : ", model.best_params_)

print("best_score_ : ", model.best_score_)

print("model.score : ", model.score(x_test, y_test))

###################################### 

#4. 평가
y_predict = model.predict(x_test)
print("accuracy_score", round(accuracy_score(y_test, y_predict),4))

# y_pred_best = model.best_estimator_.__prepare__(x_test)
# print('최적 튠 ACC : ', accuracy_score(y_test, y_pred_best))