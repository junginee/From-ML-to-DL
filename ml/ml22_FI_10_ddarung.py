# 실습
# feature importance가 전체 중요도해서 하위 20~25% 컬럼들을 제거하여
# 데이터셋 재구성후
# 각 모델별로 돌려서 결과 도출
# 기존 모델결과와 비교

from multiprocessing import Pipe
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터
path = './_data/ddarung/'
train_set = pd.read_csv(path + 'train.csv', 
                        index_col=0) 


test_set = pd.read_csv(path + 'test.csv', 
                       index_col=0)


train_set =  train_set.dropna()


test_set = test_set.fillna(test_set.mean())


x = train_set.drop(['count'], axis=1) 
print('before',x.shape) #before (1328, 9)

y = train_set['count']

x = np.delete(x,[2,8],axis=1) #after (20640, 6)
print('after',x.shape) 

x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.8,shuffle=True, random_state=1234)

# 2. 모델구성 
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor

model1 = DecisionTreeRegressor()
model2=  RandomForestRegressor()
model3 = GradientBoostingRegressor()
model4 = XGBRegressor()


#3. 훈련
model1.fit(x_train, y_train) 
model2.fit(x_train, y_train) 
model3.fit(x_train, y_train) 
model4.fit(x_train, y_train) 

print(model1,':' ,model1.feature_importances_)  
print(model2,':', model2.feature_importances_) 
print(model3,':',model3.feature_importances_) 
print(model4,':',model4.feature_importances_) 


#4. 평가, 예측
result1 = model1.score(x_test, y_test) 
result2 = model2.score(x_test, y_test) 
result3 = model3.score(x_test, y_test) 
result4 = model4.score(x_test, y_test) 

from sklearn.metrics import accuracy_score, r2_score
y_predict1 = model1.predict(x_test)
r2_1 = r2_score(y_test, y_predict1) 

y_predict2 = model2.predict(x_test)
r2_2 = r2_score(y_test, y_predict2) 

y_predict3 = model3.predict(x_test)
r2_3 = r2_score(y_test, y_predict3) 

y_predict4 = model4.predict(x_test)
r2_4 = r2_score(y_test, y_predict4) 

print(model1,':' ,round(r2_1,4))  
print(model2,':', round(r2_2,4))  
print(model3,':', round(r2_3,4)) 
print(model4,':', round(r2_4,4))  

#column drop 전
# DecisionTreeRegressor() : 0.6107
# RandomForestRegressor() : 0.8026
# GradientBoostingRegressor() : 0.7834
# XGBRegressor : 0.7801

#column drop 후
