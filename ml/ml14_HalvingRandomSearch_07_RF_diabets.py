import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, StratifiedKFold,RandomizedSearchCV
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score
from tqdm import tqdm_notebook
from sklearn.experimental import enable_halving_search_cv   
from sklearn.model_selection import HalvingRandomSearchCV

#1. 데이터

path = './_data/kaggle_titanic/' 
train_set = pd.read_csv(path + 'train.csv',
                        index_col=0) 
test_set = pd.read_csv(path + 'test.csv',
                       index_col=0)

# print(train_set)
# print(train_set.shape)

# print(train_set.columns)
# print(train_set.info()) 
# print(train_set.describe())

print(test_set)
print(test_set.shape)

# 결측치 처리
print(train_set.isnull().sum()) 
train_set = train_set.fillna(train_set.median())
print(test_set.isnull().sum())

drop_cols = ['Cabin']
train_set.drop(drop_cols, axis = 1, inplace =True)
test_set = test_set.fillna(test_set.mean())
train_set['Embarked'].fillna('S')
train_set = train_set.fillna(train_set.mean())

# print(train_set) 
# print(train_set.isnull().sum())

test_set.drop(drop_cols, axis = 1, inplace =True)
cols = ['Name','Sex','Ticket','Embarked']
for col in tqdm_notebook(cols):
    le = LabelEncoder()
    train_set[col]=le.fit_transform(train_set[col])
    test_set[col]=le.fit_transform(test_set[col])
    
x = train_set.drop(['Survived'],axis=1) #axis는 컬럼 
# print(x) #(891, 9)
y = train_set['Survived']
# print(y.shape) #(891,)

gender_submission = pd.read_csv(path + 'gender_submission.csv',
                       index_col=0)

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
from sklearn.linear_model import Perceptron, LogisticRegression # LogisticRegression 분류 모델 사용
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier # 가지치기 형식으로 결과값 도출, 분류형식
from sklearn.ensemble import RandomForestClassifier # DecisionTreeClassifier가 ensemble 엮여있는게 random으로 

# model = SVC(C=1, kernel='linear', degree=3)
model = HalvingRandomSearchCV(RandomForestClassifier(),parameters, cv=kfold, verbose=1,                              
                     refit=True, n_jobs=-1)                          
                                                                    
#3. 컴파일, 훈련
import time
start = time.time()
model.fit(x_train, y_train)
end = time.time()

print("최적의 매개변수 : ", model.best_estimator_)  
# 최적의 매개변수 :  SVC(C=100, gamma=0.001)
print("최적의 파라미터 : ", model.best_params_)
# 최적의 파라미터 :  {'C': 100, 'gamma': 0.001, 'kernel': 'rbf'}

print("best_score_ : ", model.best_score_)      
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
# 최적의 매개변수 :  RandomForestClassifier(min_samples_leaf=3, min_samples_split=10)
# 최적의 파라미터 :  {'min_samples_leaf': 3, 'min_samples_split': 10}
# best_score_ :  0.841327686398109
# model.score :  0.8324022346368715
# accuracy_score :  0.8324022346368715
# 최적 튠 ACC :  0.8324022346368715
# 걸린시간 :  19.5078

# RandomizedSearchCV
# 최적의 매개변수 :  RandomForestClassifier(min_samples_split=5, n_jobs=2)
# 최적의 파라미터 :  {'n_jobs': 2, 'min_samples_split': 5}
# best_score_ :  0.8385107849896583
# model.score :  0.8100558659217877
# accuracy_score :  0.8100558659217877
# 최적 튠 ACC :  0.8100558659217877
# 걸린시간 :  3.5522

# HalvingGridSearchCV
# 최적의 매개변수 :  RandomForestClassifier(max_depth=12)
# 최적의 파라미터 :  {'max_depth': 12, 'n_estimators': 100}
# best_score_ :  0.8138629283489097
# model.score :  0.8100558659217877
# accuracy_score :  0.8100558659217877
# 최적 튠 ACC :  0.8100558659217877
# 걸린시간 :  30.7935

# HalvingRandomSearchCV
# 최적의 매개변수 :  RandomForestClassifier(min_samples_split=10, n_jobs=2)
# 최적의 파라미터 :  {'n_jobs': 2, 'min_samples_split': 10}
# best_score_ :  0.8286950501903773
# model.score :  0.8379888268156425
# accuracy_score :  0.8379888268156425
# 최적 튠 ACC :  0.8379888268156425
# 걸린시간 :  8.7398
