parameters =[
    {'n_estimators': [100,200]}, # epochs 의 수를 조정하는 옵션
    {'max_depth': [6,8,10,12]}, # 최대 깊이 조정하는 옵션
    {'min_samples_leaf': [3,5,7,10]}, # 리프 노드의 최소 샘플 수 조정하는 옵션
    {'min_samples_split': [2,3,5,10]}, # 분할 노드의 최소 샘플 수 조정하는 옵션
    {'n_jobs': [-1,2,4]} # cpu 수 (병렬 처리 수) 조정하는 옵션 -- 성능이 아닌 속도에만 영향을 줌
]

# m09_03

from unittest import result
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, StratifiedKFold 
from sklearn.metrics import accuracy_score, r2_score

datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x)
print(y)
print(x.shape, y.shape) #(442, 10) (442,) Number of Instances: 442, Number of Attributes 10

print(datasets.feature_names)
print(datasets.DESCR)


x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.8, shuffle=True, random_state=66)

n_splits=5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=66)

parameters = [
    {'n_estimators': [100,200], 'max_depth': [6,8,10,12], 'min_samples_leaf': [3,5,7,10]},          # 32번
    {'max_depth': [6,8,10,12], 'min_samples_leaf': [3,5,7,10], 'min_samples_split': [2,3,5,10]},    # 64번
    {'min_samples_leaf': [3,5,7,10], 'min_samples_split': [2,3,5,10], 'n_jobs': [-1,2,4]}           # 48번 = 총 144번
]


#2. 모델구성
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor 
model = RandomizedSearchCV(RandomForestRegressor(), parameters, cv=kfold, verbose=1,
                     refit=True, n_jobs=-1)


#3. 컴파일, 훈련
import time
start = time.time()
model.fit(x_train, y_train)
end = time.time()

print("최적의 매개변수 : ", model.best_estimator_) 
print("최적의 파라미터  : ", model.best_params_)
print('best_score : ', model.best_score_)
print('model.score : ', model.score(x_test, y_test))

y_predict = model.predict(x_test)
print('r2_score : ', r2_score(y_test, y_predict))

y_pred_best = model.best_estimator_.predict(x_test)
print('최적 튠 ACC :', r2_score(y_test, y_pred_best))
print('걸린시간 :', np.round(end-start, 4), "초")


# RandomizedSearchCV
# 최적의 매개변수 :  RandomForestRegressor(max_depth=12, min_samples_leaf=10, min_samples_split=10)     
# 최적의 파라미터  :  {'min_samples_split': 10, 'min_samples_leaf': 10, 'max_depth': 12}
# best_score :  0.47497523752934995
# model.score :  0.4071301230026233
# r2_score :  0.4071301230026233
# 최적 튠 ACC : 0.4071301230026233
# 걸린시간 : 5.8177 초


# GridSearchCV
# Fitting 5 folds for each of 144 candidates, totalling 720 fits
# 최적의 매개변수 :  RandomForestRegressor(max_depth=6, min_samples_leaf=10)
# 최적의 파라미터  :  {'max_depth': 6, 'min_samples_leaf': 10, 'n_estimators': 100}
# best_score :  0.4829903742012565
# model.score :  0.40448589011887204
# r2_score :  0.40448589011887204
# 최적 튠 ACC : 0.40448589011887204
# 걸린시간 : 18.1175 초
