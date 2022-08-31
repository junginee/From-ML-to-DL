import numpy as np
import pandas as pd 
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import LinearSVC, SVC
from sklearn. ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn. model_selection import HalvingGridSearchCV, HalvingRandomSearchCV


#1. 데이터
path = './_data/kaggle_titanic/' 
train_set = pd.read_csv(path + 'train.csv',
                        index_col=0)
test_set = pd.read_csv(path + 'test.csv', 
                       index_col=0)

train_set = train_set.fillna(train_set.median())


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

gender_submission = pd.read_csv(path + 'gender_submission.csv',
                       index_col=0)


x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.8,shuffle=True, random_state=1234)



#2. 모델구성
pipe = Pipeline([('minmax', MinMaxScaler()), ('RF', RandomForestClassifier())], verbose=1)                                             

parameters = [
        {'RF__n_estimators':[100,200], 'RF__max_depth':[6,8,10,12], 'RF__min_samples_split':[2,3,5,10]},
        {'RF__n_estimators':[100,200], 'RF__min_samples_leaf':[3,5,7,10], 'RF__min_samples_split':[2,3,5,10]},
]

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle = True, random_state=66)
                                           
#3. 훈련
model = GridSearchCV(pipe, parameters, cv= kfold, verbose=1)
model.fit(x_train, y_train) 

#4. 평가, 예측
result = model.score(x_test, y_test)
print('[kaggle_titanic]')
print('model.score:', result)        

# [kaggle_titanic]
# model.score: 0.8436
