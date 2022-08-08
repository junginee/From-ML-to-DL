import numpy as np
from sklearn.datasets import load_diabetes
from tensorflow.python.keras.utils.generic_utils import default
from xgboost import XGBClassifier, XGBRegressor
import pandas as pd
from sklearn.model_selection import train_test_split

path = './_data/kaggle_bike/'

train = pd.read_csv(path+'train.csv')
test_file = pd.read_csv(path+'test.csv')
submit_file = pd.read_csv(path+'sampleSubmission.csv')

train = pd.read_csv(path + 'train.csv')
test_file = pd.read_csv(path + 'test.csv')  
submit_file = pd.read_csv(path + 'sampleSubmission.csv')

x = train.drop(['datetime', 'casual','registered', 'count'], axis=1) 
test_file = test_file.drop(['datetime'], axis=1)
y = train['count']

x = np.delete(x,[1,3,4],axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

#2) 모델구성 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

model1 = DecisionTreeRegressor(max_depth=5)
model2= RandomForestRegressor(max_depth=5)
model3 = XGBRegressor()
model4 = GradientBoostingRegressor()

#3) 훈련

model1.fit(x_train, y_train) 
model2.fit(x_train, y_train) 
model3.fit(x_train, y_train) 
model4.fit(x_train, y_train) 

#4) 평가, 예측
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

print("accuracy: ", r2_1)  
print("accuracy: ", r2_2)  
print("accuracy: ", r2_3)  
print("accuracy: ", r2_4)  

print(model1.feature_importances_)  
print(model2.feature_importances_) 
print(model3.feature_importances_) 
print(model4.feature_importances_)  