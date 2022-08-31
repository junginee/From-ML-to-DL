import numpy as np
import time
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, r2_score
                        
#1.데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size=0.2,
                                                    shuffle=True,
                                                    random_state=66)
n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle = True, random_state=66)


parameters = [
        {'n_estimators':[100,200], 'max_depth':[6,8,10,12], 'min_samples_split':[2,3,5,10]},
        {'n_estimators':[100,200], 'min_samples_leaf':[3,5,7,10], 'min_samples_split':[2,3,5,10]},
]                                                                         # 총 42번


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
# 최적의 매개변수 :  RandomForestRegressor(max_depth=12, n_estimators=200)
print("최적의 파라미터 : ", model.best_params_)
# 최적의 파라미터 :  {'max_depth': 12, 'min_samples_split': 2, 'n_estimators': 200}
print("best_score_ : ", model.best_score_)
# best_score_ :  0.8358261014826864
print("model.score : ", model.score(x_test, y_test))
# model.score :  0.9207262926190887

#4. 평가
y_predict = model.predict(x_test)
print("r2_score", round(r2_score(y_test, y_predict),4))
# r2_score 0.9207

# y_pred_best = model.best_estimator_.__prepare__(x_test)
# print('최적 튠 ACC : ', accuracy_score(y_test, y_pred_best))
