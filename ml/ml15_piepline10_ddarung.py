import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC, SVC
from sklearn. ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import make_pipeline

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

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.8,shuffle=True, random_state=1234)



#2. 모델구성
#model = SVC()
model = make_pipeline(MinMaxScaler(),RandomForestRegressor())  
                                           
                                             
#3. 훈련
model.fit(x_train, y_train)  


#4. 평가, 예측
result = model.score(x_test, y_test)
print('ddarung')
print('model.score:', round(result,4))    

# ddarung
# model.score: 0.8092                                         
