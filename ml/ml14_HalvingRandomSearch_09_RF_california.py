import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, r2_score
from sklearn.experimental import enable_halving_search_cv   
from sklearn.model_selection import HalvingRandomSearchCV

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
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
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron, LogisticRegression # LogisticRegression 분류 모델 사용
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier # 가지치기 형식으로 결과값 도출, 분류형식
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor # DecisionTreeClassifier가 ensemble 엮여있는게 random으로 

# model = SVC(C=1, kernel='linear', degree=3)
model = HalvingRandomSearchCV(RandomForestRegressor(),parameters, cv=kfold, verbose=1,             # 42 * 5 = 210
                     refit=True, n_jobs=-1)                             # n_jobs는 cpu 사용 갯수
                                                                        # refit=True 최적의 값을 찾아서 저장 후 모델 학습
                                                                    
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
# 최적의 매개변수 :  RandomForestRegressor()
# 최적의 파라미터 :  {'min_samples_split': 2}
# best_score_ :  0.8048220332384407
# model.score :  0.8041775120334725
# r2_score :  0.8041775120334725
# 최적 튠 R2 :  0.8041775120334725
# 걸린시간 :  574.59

# RandomizedSearchCV
# 최적의 매개변수 :  RandomForestRegressor(min_samples_split=3, n_jobs=2)
# 최적의 파라미터 :  {'n_jobs': 2, 'min_samples_split': 3}
# best_score_ :  0.8043379535705342
# model.score :  0.8018006083210785
# r2_score :  0.8018006083210785
# 최적 튠 R2 :  0.8018006083210785
# 걸린시간 :  89.4272

# HalvingGridSearchCV
# 최적의 매개변수 :  RandomForestRegressor(max_depth=16, n_estimators=400)
# 최적의 파라미터 :  {'max_depth': 16, 'n_estimators': 400}
# best_score_ :  0.8033120290339344
# model.score :  0.8029365810932574
# r2_score :  0.8029365810932574
# 최적 튠 R2 :  0.8029365810932574
# 걸린시간 :  115.8524

# HalvingRandomSearchCV
# 최적의 매개변수 :  RandomForestRegressor(max_depth=10, min_samples_leaf=5)
# 최적의 파라미터 :  {'min_samples_leaf': 5, 'max_depth': 10}
# best_score_ :  0.689146114266855
# model.score :  0.7740772315861243
# r2_score :  0.7740772315861243
# 최적 튠 R2 :  0.7740772315861243
# 걸린시간 :  25.749