import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn. datasets import load_digits
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
from sklearn.svm import LinearSVC

#1. 데이터
datasets = load_digits()
x = datasets.data
y= datasets.target

print(x.shape, y.shape)
print(np.unique(y)) 

import tensorflow as tf
tf.random.set_seed(66)

x_train, x_test, y_train, y_test = train_test_split( x, y, train_size = 0.8, shuffle=True, random_state=68 )

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)
print(np.min(x_train)) 
print(np.max(x_train))

#2. 모델구성
model = LinearSVC()  

#3. 컴파일, 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test, y_test)
print("결과 acc : ", round(results,3)) 

from sklearn.metrics import r2_score, accuracy_score  
y_predict = model.predict(x_test)

acc= accuracy_score(y_test, y_predict)
print('acc스코어 : ', round(acc,3))  

# 결과 acc :  0.964
# acc스코어 :  0.964
