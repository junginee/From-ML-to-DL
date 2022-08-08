# 실습
# feature importance가 전체 중요도해서 하위 20~25% 컬럼들을 제거하여
# 데이터셋 재구성후
# 각 모델별로 돌려서 결과 도출
# 기존 모델결과와 비교

import numpy as np
from sklearn. datasets import fetch_covtype 
from sklearn.model_selection import train_test_split
import pandas as pd

# 1. 데이터

datasets = fetch_covtype()
x = datasets.data
y = datasets.target
print('before',x.shape) #before (581012, 54)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

# 2. 모델구성 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

model1 = DecisionTreeClassifier()
model2= RandomForestClassifier()
model3 = XGBClassifier()
model4 = GradientBoostingClassifier()


#3. 훈련
model1.fit(x_train, y_train) 
model2.fit(x_train, y_train) 
model3.fit(x_train, y_train) 
model4.fit(x_train, y_train) 

print(model1,':' ,model1.feature_importances_)  
print(model2,':', model2.feature_importances_) 
print(model3,':',model3.feature_importances_) 
print(model4,':',model4.feature_importances_) 

'''

x = np.delete(x,[1,2,4,5,7,8],axis=1) 
print('after',x.shape) #after (178, 7)

#4. 평가, 예측
result1 = model1.score(x_test, y_test) 
result2 = model2.score(x_test, y_test) 
result3 = model3.score(x_test, y_test) 
result4 = model4.score(x_test, y_test) 

from sklearn.metrics import accuracy_score
y_predict1 = model1.predict(x_test)
acc1 = accuracy_score(y_test, y_predict1) 

y_predict2 = model2.predict(x_test)
acc2 = accuracy_score(y_test, y_predict2) 

y_predict3 = model3.predict(x_test)
acc3 = accuracy_score(y_test, y_predict3) 

y_predict4 = model4.predict(x_test)
acc4 = accuracy_score(y_test, y_predict4) 

print(model1,':' ,round(acc1,4))  #DecisionTreeClassifier() : 0.9444
print(model2,':', round(acc2,4))  #RandomForestClassifier() : 1.0
print(model3,':', round(acc3,4))  #XGBClassifier: 1.0
print(model4,':', round(acc4,4))  #GradientBoostingClassifier() : 0.9722

'''