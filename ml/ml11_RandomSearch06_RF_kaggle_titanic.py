import numpy as np
import pandas as pd 
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings("ignore")



#1.데이터

path = './_data/kaggle_titanic/' 
train_set = pd.read_csv(path + 'train.csv',
                        index_col=0)
test_set = pd.read_csv(path + 'test.csv', 
                       index_col=0)

# print(test_set) 
# print(train_set.isnull().sum()) 

train_set = train_set.fillna(train_set.median())

# print(test_set.isnull().sum())


drop_cols = ['Cabin']
train_set.drop(drop_cols, axis = 1, inplace =True)
test_set = test_set.fillna(test_set.mean())
train_set['Embarked'].fillna('S')
train_set = train_set.fillna(train_set.mean())

print("===============================")


print(train_set) 
print(train_set.isnull().sum())

test_set.drop(drop_cols, axis = 1, inplace =True)
cols = ['Name','Sex','Ticket','Embarked']

from tqdm import tqdm_notebook
for col in tqdm_notebook(cols):
    le = LabelEncoder()
    train_set[col]=le.fit_transform(train_set[col])
    test_set[col]=le.fit_transform(test_set[col])
    
x = train_set.drop(['Survived'],axis=1) 
print(x) #(891, 9)

y = train_set['Survived']
print(y.shape) #(891,)

gender_submission = pd.read_csv(path +
                       index_col=0)


x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.91,shuffle=True ,random_state=100)

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)
test_set = scaler.transform(test_set)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle = True, random_state=66)                                                  

parameters = [
        {'n_estimators':[100,200], 'max_depth':[6,8,10,12]},
        {'n_estimators':[100,200], 'min_samples_leaf':[3,5,7,10], 'min_samples_split':[2,3,5,10]},
] 

#2. 모델구성
model = RandomForestClassifier(max_depth=10)        
# model = GridSearchCV(RandomForestClassifier(), parameters, cv=kfold, verbose=1,          
#                       refit=True, n_jobs=1)    
                                                                        
#3. 컴파일, 훈련
model.fit(x_train, y_train)

####### GridSearchCV 탐색결과 #######
# print("최적의 매개변수 : ", model.best_estimator_)
# 최적의 매개변수 :  RandomForestClassifier(max_depth=10)
# print("최적의 파라미터 : ", model.best_params_)
# 최적의 파라미터 :  {'max_depth': 10, 'n_estimators': 100}
# print("best_score_ : ", model.best_score_)
# best_score_ :  0.8395061728395061
# print("model.score : ", model.score(x_test, y_test))
# model.score :  0.8148148148148148
###################################### 

#4. 평가
y_predict = model.predict(x_test)
print("accuracy_score", round(accuracy_score(y_test, y_predict),4))

# accuracy_score 0.8272
