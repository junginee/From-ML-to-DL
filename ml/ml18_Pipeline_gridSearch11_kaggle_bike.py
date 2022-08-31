
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn. model_selection import HalvingGridSearchCV, HalvingRandomSearchCV

#1. 데이터
path = './_data/kaggle_bike/'
train_set = pd.read_csv(path + 'train.csv')
test_set = pd.read_csv(path + 'test.csv')
train_set['datetime'] = pd.to_datetime(train_set['datetime']) 
train_set['year'] = train_set['datetime'].dt.year  
train_set['month'] = train_set['datetime'].dt.month
train_set['day'] = train_set['datetime'].dt.day
train_set['hour'] = train_set['datetime'].dt.hour
train_set.drop(['datetime', 'day', 'year'], inplace=True, axis=1)
train_set['month'] = train_set['month'].astype('category')
train_set['hour'] = train_set['hour'].astype('category')
train_set = pd.get_dummies(train_set, columns=['season','weather'])
train_set.drop(['casual', 'registered'], inplace=True, axis=1)
train_set.drop('atemp', inplace=True, axis=1)

test_set['datetime'] = pd.to_datetime(test_set['datetime'])
test_set['month'] = test_set['datetime'].dt.month
test_set['hour'] = test_set['datetime'].dt.hour
test_set['month'] = test_set['month'].astype('category')
test_set['hour'] = test_set['hour'].astype('category')
test_set = pd.get_dummies(test_set, columns=['season','weather'])
drop_feature = ['datetime', 'atemp']
test_set.drop(drop_feature, inplace=True, axis=1)

x = train_set.drop(['count'], axis=1)
y = train_set['count']


x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.8,shuffle=True, random_state=1234)



#2. 모델구성
from sklearn.svm import LinearSVC, SVC
from sklearn. ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import make_pipeline, Pipeline


# model = SVC()
# model = make_pipeline(MinMaxScaler(),PCA(),RandomForestClassifier())
pipe = Pipeline([('minmax', MinMaxScaler()), ('RF', RandomForestRegressor())], verbose=1)                                           

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
print('kaggle_bike')
print('model.score:', round(result,4))    

# kaggle_bike
# model.score: 0.8525
