import numpy as np
from sklearn.datasets import load_iris 
from sklearn.model_selection import cross_val_predict, train_test_split,KFold, cross_val_score 
from sklearn.utils import all_estimators
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score 
import tensorflow as tf
tf.random.set_seed(66)

#1. 데이터
datasets = load_iris()
x = datasets['data']
y = datasets.target


print(x.shape, y.shape) 
print("y의 라벨값(y의 고유값)", np.unique(y)) #y의 라벨값(y의 고유값) [0 1 2]


x_train, x_test, y_train, y_test = train_test_split( x, y, train_size = 0.8, shuffle=True, random_state=72 )

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle = True, random_state=66)

#2. 모델구성
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import Perceptron, LogisticRegression  
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn. ensemble import RandomForestClassifier

model = SVC()
#3,4. 컴파일, 훈련, 평가, 예측

# model.fit(x_train, y_train)
scores = cross_val_score(model,x_train,y_train,cv=kfold)
print('ACC : ',scores,'\ncross_val_score :', round(np.mean(scores),4))

y_predict = cross_val_predict(model, x_test, y_test, cv=kfold)
print(y_predict)
print(y_test)
acc = accuracy_score(y_test, y_predict)
print("cross_val_predict ACC : ", round(acc,4))

# ACC :  [0.95833333 0.91666667 0.95833333 0.95833333 1.        ] 
# cross_val_score : 0.9583
# [0 1 2 2 2 1 1 2 0 0 0 0 1 0 0 1 2 0 0 0 2 2 0 1 1 2 0 2 2 0]
# [0 1 2 2 2 1 1 2 0 0 0 0 1 0 0 1 2 0 0 0 1 2 0 1 1 2 0 2 2 0]
# cross_val_predict ACC :  0.9667
