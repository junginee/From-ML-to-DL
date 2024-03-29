from multiprocessing import Pipe
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import LinearSVC, SVC
from sklearn. ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import make_pipeline, Pipeline

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
        train_size=0.8,shuffle=True, random_state=1234)


#2. 모델구성
# model = SVC()
# model = make_pipeline(MinMaxScaler(),PCA(),RandomForestClassifier())  # piepline을 통해 순서대로 이동
pipe = Pipeline([('minmax', MinMaxScaler()), ('RF', RandomForestRegressor())], verbose=1)                                             

parameters = [
        {'RF__n_estimators':[100,200], 'RF__max_depth':[6,8,10,12], 'RF__min_samples_split':[2,3,5,10]},
        {'RF__n_estimators':[100,200], 'RF__min_samples_leaf':[3,5,7,10], 'RF__min_samples_split':[2,3,5,10]},
]

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle = True, random_state=66)

                                            
#3. 훈련
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn. model_selection import HalvingGridSearchCV, HalvingRandomSearchCV

model = GridSearchCV(pipe, parameters, cv= kfold, verbose=1)
model.fit(x_train, y_train) 

#4. 평가, 예측
result = model.score(x_test, y_test)
print('ddaung')
print('model.score:', round(result,4))    

# ddaung
# model.score: 0.8057
