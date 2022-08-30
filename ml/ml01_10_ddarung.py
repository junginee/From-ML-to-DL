import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR

#1.데이터
path = './_data/ddarung/'
train_set = pd.read_csv(path + 'train.csv', index_col =0)]
test_set = pd.read_csv(path + 'test.csv', index_col =0) 
train_set = train_set.dropna() 

x = train_set.drop(['count'], axis = 1)
y = train_set['count'] 

x_train,x_test,y_train, y_test = train_test_split(x,y, train_size=0.7, shuffle=True, random_state=72)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)
test_set = scaler.transform(test_set)

#2.모델구성
model = LinearSVR() 

#3. 컴파일, 훈련     
model.fit(x_train, y_train)    

#4. 평가, 예측
results = model.score(x_test, y_test)
print("결과 : ", round(results,3)) 

# 결과 :  0.551 
