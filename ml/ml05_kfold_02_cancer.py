
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, KFold, cross_val_score 
from sklearn.metrics import r2_score, accuracy_score
from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore")

#1. 데이터
datasets = load_breast_cancer()

x = datasets.data   
y = datasets.target
print(x.shape, y.shape) 


x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.7,shuffle=True, random_state=72)
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)
print(np.min(x_train)) #0.0 
print(np.max(x_train)) #1.0                                 
                
n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle = True, random_state=66)                                                  


#2. 모델구성
model = SVC() 
                                                                        
#3,4. 컴파일, 훈련, 평가, 예측

scores = cross_val_score(model,x,y,cv=kfold)
print('ACC : ',scores,'\ncross_val_score :', round(np.mean(scores),4))

# ACC :  [0.89473684 0.92982456 0.89473684 0.92105263 0.96460177] 
# cross_val_score : 0.921