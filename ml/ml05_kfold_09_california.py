import numpy as np
from sklearn.model_selection import train_test_split,KFold, cross_val_score 
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import RobustScaler
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.linear_model import Perceptron, LinearRegression

datasets = fetch_california_housing() 
x = datasets.data 
y = datasets.target 
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, shuffle=True, random_state=5)


scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)
print(np.min(x_train)) #0.0 
print(np.max(x_train)) #1.0

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle = True, random_state=66)

#2. 모델구성
model = LinearRegression()
                

#3,4. 컴파일, 훈련, 평가, 예측
scores = cross_val_score(model,x,y,cv=kfold)
print('ACC : ',scores,'\ncross_val_score :', round(np.mean(scores),4))

# ACC :  [0.61614066 0.59250746 0.59486463 0.5981755  0.60724957] 
# cross_val_score : 0.6018
