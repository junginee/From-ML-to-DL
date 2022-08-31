
import numpy as np
import tensorflow as tf
from sklearn. datasets import fetch_covtype 
from sklearn.model_selection import train_test_split,KFold, cross_val_score 
from sklearn.metrics import accuracy_score 
from sklearn.svm import LinearSVC
import warnings
warnings.filterwarnings("ignore")

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y= datasets.target

print(datasets.feature_names)
print(datasets.DESCR)
print(x.shape, y.shape) #(581012, 54) (581012,)
print(np.unique(y, return_counts = True)) # y :[1 2 3 4 5 6 7]  / return_counts :[211840, 283301,  35754,   2747,   9493,  17367,  20510]

x_train, x_test, y_train, y_test = train_test_split( x, y, train_size = 0.8, shuffle=True, random_state=68 )

                
n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle = True, random_state=66)  

#2. 모델구성
model = LinearSVC()  

#3,4. 컴파일, 훈련, 평가, 예측

scores = cross_val_score(model,x,y,cv=kfold)
print('ACC : ',scores,'\ncross_val_score :', round(np.mean(scores),4))


