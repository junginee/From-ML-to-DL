import numpy as np
from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, HalvingRandomSearchCV
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from sklearn. ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, accuracy_score

datasets =load_diabetes()
x = datasets.data
y = datasets.target
x_train,x_test,y_train, y_test = train_test_split(x,y, train_size=0.75, shuffle=True, random_state=72)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)
print(np.min(x_train)) #0.0 
print(np.max(x_train)) #1.0

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle = True, random_state=120)    

parameters = [
        {'n_estimators':[100,200], 'max_depth':[6,8,10,12],'min_samples_split':[2,3,5,10]},
        {'n_estimators':[100,200], 'min_samples_leaf':[3,5,7,10], 'min_samples_split':[2,3,5,10]},
] 

#2. 모델구성
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron, LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

                     
model = HalvingRandomSearchCV(RandomForestRegressor(), parameters, cv=kfold, verbose=1,          
                     refit=True, n_jobs=1)                             
                                                                           


#3. 컴파일, 훈련
model.fit(x_train, y_train)

####### GridSearchCV 탐색결과 #######
# print("최적의 매개변수 : ", model.best_estimator_)
# 최적의 매개변수 :  RandomForestRegressor(min_samples_leaf=10, min_samples_split=3)
# print("최적의 파라미터 : ", model.best_params_)
# 최적의 파라미터 :  {'min_samples_leaf': 10, 'min_samples_split': 3, 'n_estimators': 100}
# print("best_score_ : ", model.best_score_)
# best_score_ :  0.4353879532270981
# print("model.score : ", model.score(x_test, y_test))
# model.score :  0.5845886025214724
###################################### 

### HalvingRandomSearchCV 탐색결과 ###
print("최적의 매개변수 : ", model.best_estimator_)
# 최적의 매개변수 :  RandomForestClassifier(max_depth=12, min_samples_split=3)
print("최적의 파라미터 : ", model.best_params_)
# 최적의 파라미터 :  {'n_estimators': 100, 'min_samples_split': 3, 'max_depth': 12}
print("best_score_ : ", model.best_score_)
# best_score_ :  0.9663492063492063
print("model.score : ", model.score(x_test, y_test))
# model.score :  0.9532163742690059
###################################### 

#4. 평가
y_predict = model.predict(x_test)
print("r2_score", round(r2_score(y_test, y_predict),4))
# r2_score 0.5834

# y_pred_best = model.best_estimator_.__prepare__(x_test)
# print('최적 튠 ACC : ', accuracy_score(y_test, y_pred_best))import numpy as np
