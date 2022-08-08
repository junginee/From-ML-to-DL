# 실습
# feature importance가 전체 중요도해서 하위 20~25% 컬럼들을 제거하여
# 데이터셋 재구성후
# 각 모델별로 돌려서 결과 도출
# 기존 모델결과와 비교

import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import pandas as pd

# 1. 데이터

datasets = load_boston()
x = datasets.data
y = datasets.target
print('before',x.shape) #before (506, 13)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

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

x = np.delete(x,[1,3],axis=1) #after (442, 8)
print('after',x.shape) 

#4. 평가, 예측
result1 = model1.score(x_test, y_test) 
result2 = model2.score(x_test, y_test) 
result3 = model3.score(x_test, y_test) 
result4 = model4.score(x_test, y_test) 

from sklearn.metrics import accuracy_score, r2_score
y_predict1 = model1.predict(x_test)
acc1 = r2_score(y_test, y_predict1) 

y_predict2 = model2.predict(x_test)
acc2 = r2_score(y_test, y_predict2) 

y_predict3 = model3.predict(x_test)
acc3 = r2_score(y_test, y_predict3) 

y_predict4 = model4.predict(x_test)
acc4 = r2_score(y_test, y_predict4) 

print(model1,':' ,round(acc1,4))  #DecisionTreeRegressor() : 0.7251
print(model2,':', round(acc2,4))  #RandomForestRegressor() : 0.9224
print(model3,':', round(acc3,4))  #GradientBoostingRegressor() : 0.945
print(model4,':', round(acc4,4))  #XGBRegressor() : : 0.9221

