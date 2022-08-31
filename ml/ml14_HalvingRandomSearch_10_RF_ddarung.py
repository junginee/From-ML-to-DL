import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split,RandomizedSearchCV
from sklearn.metrics import accuracy_score, r2_score
from sklearn.experimental import enable_halving_search_cv   
from sklearn.model_selection import HalvingRandomSearchCV
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
path = './_data/ddarung/'
train_set = pd.read_csv(path + 'train.csv', 
                        index_col=0) 

test_set = pd.read_csv(path + 'test.csv', 
                       index_col=0)

train_set =  train_set.dropna()
test_set = test_set.fillna(test_set.mean())


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
# 최적의 매개변수 :  RandomForestRegressor(max_depth=12)
# 최적의 파라미터 :  {'max_depth': 12, 'n_estimators': 100}
# best_score_ :  0.7673526022221917
# model.score :  0.7701840050833797
# r2_score :  0.7701840050833797
# 최적 튠 R2 :  0.7701840050833797
# 걸린시간 :  32.5641

# RandomizedSearchCV
# 최적의 매개변수 :  RandomForestRegressor(min_samples_split=5, n_jobs=6)
# 최적의 파라미터 :  {'n_jobs': 6, 'min_samples_split': 5}
# best_score_ :  0.7630833611064008
# model.score :  0.7729084838020452
# r2_score :  0.7729084838020452
# 최적 튠 R2 :  0.7729084838020452
# 걸린시간 :  4.1015

# HalvingGridSearchCV
# 최적의 매개변수 :  RandomForestRegressor(n_jobs=4)
# 최적의 파라미터 :  {'min_samples_split': 2, 'n_jobs': 4}
# best_score_ :  0.761292045510392
# model.score :  0.7754452504088395
# r2_score :  0.7754452504088395
# 최적 튠 R2 :  0.7754452504088395
# 걸린시간 :  23.6543

# HalvingRandomSearchCV
# 최적의 매개변수 :  RandomForestRegressor(max_depth=14, min_samples_leaf=3)
# 최적의 파라미터 :  {'min_samples_leaf': 3, 'max_depth': 14}
# best_score_ :  0.7539925780417219
# model.score :  0.7674643795951215
# r2_score :  0.7674643795951215
# 최적 튠 R2 :  0.7674643795951215
# 걸린시간 :  22.3939
