
import numpy as np
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

from sklearn.model_selection import  GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv   
from sklearn.model_selection import HalvingRandomSearchCV, HalvingGridSearchCV
from sklearn.model_selection import train_test_split, KFold, cross_val_score



#1. 데이터
datasets = load_iris()

x = datasets['data']
y = datasets.target


x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.8, shuffle=True, random_state=666)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

parameters = [
    {'n_estimators' : [100,200,300,400,500], 'max_depth' : [6,10,12,14,16]},                      
    {'max_depth' : [6, 8, 10, 12, 14], 'min_samples_leaf' : [3, 5, 7, 10, 12]},         
    {'min_samples_leaf' : [3, 5, 7, 10, 12], 'min_samples_split' : [2, 3, 5, 10, 12]},  
    {'min_samples_split' : [2, 3, 5, 10, 12]},                                     
    {'n_jobs' : [-1, 2, 4, 6],'min_samples_split' : [2, 3, 5, 10, 12]}             
]


    
#2. 모델구성
from sklearn.ensemble import RandomForestClassifier # DecisionTreeClassifier가 ensemble 엮여있는게 random으로 

# model = SVC(C=1, kernel='linear', degree=3)

#[model(1)]
model = GridSearchCV(RandomForestClassifier(), parameters, cv=kfold, verbose=1,          
                     refit=True, n_jobs=1) 

#[model(2)]
model = RandomizedSearchCV(RandomForestClassifier(), parameters, cv=kfold, verbose=1,                
                   refit=True, n_jobs=1)  

#[model(3)]
model = HalvingGridSearchCV(RandomForestClassifier(),parameters, cv=kfold, verbose=1,            
                     refit=True, n_jobs=-1)   

#[model(4)]
model = HalvingRandomSearchCV(RandomForestClassifier(),parameters, cv=kfold, verbose=1,            
                     refit=True, n_jobs=-1)                            
                                                                    

#3. 컴파일, 훈련
import time
start = time.time()
model.fit(x_train, y_train)
end = time.time()

print("최적의 매개변수 : ", model.best_estimator_)  # 가장 좋은 추정치

print("최적의 파라미터 : ", model.best_params_)

print("best_score_ : ", model.best_score_)        # 정확도

print("model.score : ", model.score(x_test, y_test))


#4. 평가, 예측
y_predict = model.predict(x_test)
print("accuracy_score : ", accuracy_score(y_test, y_predict))

y_pred_best = model.best_estimator_.predict(x_test)
print("최적 튠 ACC : ", accuracy_score(y_test,y_pred_best))

print("걸린시간 : ", round(end-start, 4))

# GridSearchCV
# 최적의 매개변수 :  RandomForestClassifier(min_samples_leaf=10, min_samples_split=5)
# 최적의 파라미터 :  {'min_samples_leaf': 10, 'min_samples_split': 5}
# best_score_ :  0.95
# model.score :  1.0
# accuracy_score :  1.0
# 최적 튠 ACC :  1.0
# 걸린시간 :  14.6592

# RandomizedSearchCV
# 최적의 매개변수 :  RandomForestClassifier(min_samples_leaf=12, min_samples_split=3)
# 최적의 파라미터 :  {'min_samples_split': 3, 'min_samples_leaf': 12}
# best_score_ :  0.95
# model.score :  1.0
# accuracy_score :  1.0
# 최적 튠 ACC :  1.0
# 걸린시간 :  2.3589

# HalvingGridSearchCV
# 최적의 매개변수 :  RandomForestClassifier(max_depth=8, min_samples_leaf=5)
# 최적의 파라미터 :  {'max_depth': 8, 'min_samples_leaf': 5}
# best_score_ :  0.9333333333333332
# model.score :  1.0
# accuracy_score :  1.0
# 최적 튠 ACC :  1.0
# 걸린시간 :  23.1831

# HalvingRandomSearchCV
# 최적의 매개변수 :  RandomForestClassifier(min_samples_split=5)
# 최적의 파라미터 :  {'min_samples_split': 5}
# best_score_ :  0.9333333333333333
# model.score :  1.0
# accuracy_score :  1.0
# 최적 튠 ACC :  1.0
