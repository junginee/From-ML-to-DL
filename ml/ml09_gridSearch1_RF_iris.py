import numpy as np
import time
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
                        
#1. 데이터
datasets = load_iris()
# print(datasets.DESCR)  #행(Instances): 150   /   열(Attributes): 4
# print(datasets.feature_names)

x = datasets['data']  # .data와 동일 
y = datasets['target']  
# print(x.shape)   # (150, 4)
# print(y.shape)   # (150,)
# print("y의 라벨값 : ", np.unique(y))  # 해당 데이터의 고유값을 출력해준다.


x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.8,
                                                    shuffle=True,
                                                    random_state=100
                                                    )

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

###실습 파라미터###
# parameters =[
#     {'n_estimators':[100,200]},
#     {'max_depth':[6,8,10,12]},
#     {'min_samples_leaf':[3,5,7,10]},
#     {'min_samples_split':[2,3,5,10]},
#     {'n_estimators':[-1,2,4]},
    
# ]
###################

parameters = [
        {'n_estimators':[100,200], 'max_depth':[6,8,10,12], 'min_samples_split':[2,3,5,10]},
        {'n_estimators':[100,200], 'min_samples_leaf':[3,5,7,10], 'min_samples_split':[2,3,5,10]},
]

#2. 모델구성
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron, LogisticRegression # LogisticRegression는 분류임
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(max_depth=6, min_samples_split=3)                               
# model = GridSearchCV(RandomForestClassifier(), parameters, cv=kfold, verbose=1,          
                    #  refit=True, n_jobs=1)   
                    
                                          
                                                                   

#3. 컴파일, 훈련
model.fit(x_train, y_train)

####### GridSearchCV 탐색결과 #######
# print("최적의 매개변수 : ", model.best_estimator_)
# 최적의 매개변수 :  RandomForestClassifier(max_depth=6, min_samples_split=3)
# print("최적의 파라미터 : ", model.best_params_)
# 최적의 파라미터 :  {'max_depth': 6, 'min_samples_split': 3, 'n_estimators': 100}
# print("best_score_ : ", model.best_score_)
# best_score_ :  0.95
# print("model.score : ", model.score(x_test, y_test))
# model.score :  0.9666666666666667
######################################

#4. 평가
y_predict = model.predict(x_test)
print("accuracy_score", accuracy_score(y_test, y_predict))
# accuracy_score 0.9666666666666667


# y_pred_best = model.best_estimator_.__prepare__(x_test)
# print('최적 튠 ACC : ', accuracy_score(y_test, y_pred_best))

