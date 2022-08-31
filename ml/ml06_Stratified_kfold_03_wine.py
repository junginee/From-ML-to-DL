import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.datasets import load_wine 
from sklearn.model_selection import train_test_split,KFold, cross_val_score ,StratifiedKFold
from sklearn.metrics import accuracy_score 
from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore")

#1. 데이터
datasets = load_wine()
x = datasets.data
y= datasets.target

print(x.shape, y.shape) #(178, 13) #(178,)
print(np.unique(y, return_counts = True)) #[0 1 2]

import tensorflow as tf
tf.random.set_seed(66)

x_train, x_test, y_train, y_test = train_test_split( x, y, train_size = 0.8, shuffle=True, random_state=120 )

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)
print(np.min(x_train)) #0.0 
print(np.max(x_train)) #1.0

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle = True, random_state=66)  

#2. 모델구성
model = SVC() 


#3,4. 컴파일, 훈련, 평가, 예측

scores = cross_val_score(model,x,y,cv=kfold)
print('ACC : ',scores,'\ncross_val_score :', round(np.mean(scores),4))

# ACC :  [0.68333333 0.61016949 0.62711864]
# cross_val_score : 0.6402
