import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.experimental import enable_halving_search_cv   
from sklearn.model_selection import HalvingRandomSearchCV

#1. 데이터
datasets = load_breast_cancer()
print(datasets.DESCR)
print(datasets.feature_names)
x = datasets['data']
y = datasets.target

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
from sklearn.ensemble import RandomForestClassifier

model = HalvingRandomSearchCV(RandomForestClassifier(),parameters, cv=kfold, verbose=1,          
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
print("accuracy_score : ", accuracy_score(y_test, y_predict))
# accuracy_score :  0.9666666666666667

y_pred_best = model.best_estimator_.predict(x_test)
print("최적 튠 ACC : ", accuracy_score(y_test,y_pred_best))
# 최적 튠 ACC :  0.9666666666666667
print("걸린시간 : ", round(end-start, 4))

# GridSearchCV
# 최적의 매개변수 :  RandomForestClassifier(min_samples_leaf=3, min_samples_split=3)
# 최적의 파라미터 :  {'min_samples_leaf': 3, 'min_samples_split': 3}
# best_score_ :  0.9670329670329672
# model.score :  0.9210526315789473
# accuracy_score :  0.9210526315789473
# 최적 튠 ACC :  0.9210526315789473
# 걸린시간 :  17.1578

# RandomizedSearchCV
# 최적의 매개변수 :  RandomForestClassifier(max_depth=12, min_samples_leaf=3)
# 최적의 파라미터 :  {'min_samples_leaf': 3, 'max_depth': 12}
# best_score_ :  0.964835164835165
# model.score :  0.9298245614035088
# accuracy_score :  0.9298245614035088
# 최적 튠 ACC :  0.9298245614035088
# 걸린시간 :  3.1246

# HalvingGridSearchCV
# 최적의 매개변수 :  RandomForestClassifier(min_samples_split=5, n_jobs=-1)
# 최적의 파라미터 :  {'min_samples_split': 5, 'n_jobs': -1}
# best_score_ :  0.95
# model.score :  0.9210526315789473
# accuracy_score :  0.9210526315789473
# 최적 튠 ACC :  0.9210526315789473
# 걸린시간 :  25.7791

# HalvingRandomSearchCV
# 최적의 매개변수 :  RandomForestClassifier(n_jobs=2)
# 최적의 파라미터 :  {'n_jobs': 2, 'min_samples_split': 2}
# best_score_ :  0.9444444444444444
# model.score :  0.9210526315789473
# accuracy_score :  0.9210526315789473
# 최적 튠 ACC :  0.9210526315789473
# 걸린시간 :  5.139
