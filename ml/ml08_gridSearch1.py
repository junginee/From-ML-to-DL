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

n_splits=5
kfold = KFold(n_splits=5, shuffle=True, random_state=101)

parameters = [
    {"C":[1, 10, 100, 1000], "kernel":["linear"], "degree":[3, 4, 5]},      # 12
    {"C":[1, 10, 100], "kernel":["rbf"], "gamma":[0.001, 0.0001]},          # 6
    {"C":[1, 10, 100, 1000], "kernel":["sigmoid"],                          # 24
    "gamma":[0.01, 0.001, 0.0001], "degree":[3, 4]}
]                                                                           # 총 42번


#2. 모델구성
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron, LogisticRegression # LogisticRegression는 분류임
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

model = SVC(C=1, kernel='linear', degree=3)                                 # cv = crossvalidation
model = GridSearchCV(SVC(), parameters, cv=kfold, verbose=1,                # 42(parameters) * 5(kfold) = 210
                     refit=True, n_jobs=1)                                  # n_jobs 코어 갯수
                                                                            # n_jobs=-1로 지정해주면 모든 코어를 다 사용하기때문에 
                                                                            # 컴퓨터는 뜨거워지겠지만, 속도는 많이 빨라진다.


#3. 컴파일, 훈련
import time
start = time.time() #훈련 전 시간
model.fit(x_train, y_train)
end = time.time() #훈련 후 시간

print("최적의 매개변수 : ", model.best_estimator_)
# 최적의 매개변수 :  SVC(C=10, kernel='linear')
print("최적의 파라미터 : ", model.best_params_)
# 최적의 파라미터 :  {'C': 10, 'degree': 3, 'kernel': 'linear'}
print("best_score_ : ", model.best_score_)
# best_score_ :  0.975
print("model.score : ", model.score(x_test, y_test))
# model.score :  0.9666666666666667

#4. 평가
y_predict = model.predict(x_test)
print("accuracy_score", accuracy_score(y_test, y_predict))
# accuracy_score 0.9666666666666667

# y_pred_best = model.best_estimator_.__prepare__(x_test)
# print('최적 튠 ACC : ', accuracy_score(y_test, y_pred_best))

print("걸린시간 :", round((end-start),3))