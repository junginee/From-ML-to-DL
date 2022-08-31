import numpy as np
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split,KFold, cross_val_score 
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


# x_train, x_test, y_train, y_test = train_test_split( x, y, train_size = 0.8, shuffle=True, random_state=68 )

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

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
scores = cross_val_score(model,x,y,cv=kfold)
print('ACC : ',scores,'\ncross_val_score :', round(np.mean(scores),4))

'''
ACC :  [0.96666667 0.96666667 1.         0.93333333 0.96666667] 
cross_val_score : 0.9667
'''

