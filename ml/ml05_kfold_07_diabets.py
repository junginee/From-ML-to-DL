import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score 
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from sklearn. ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, accuracy_score


datasets =load_diabetes()
x = datasets.data
y = datasets.target
x_train,x_test,y_train, y_test = train_test_split(x,y, train_size=0.75, shuffle=True, random_state=72)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)
print(np.min(x_train)) #0.0 
print(np.max(x_train)) #1.0

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle = True, random_state=120)    

#2.모델구성
model = RandomForestRegressor() 

                                                                        
#3,4. 컴파일, 훈련, 평가, 예측

scores = cross_val_score(model,x,y,cv=kfold)
print('ACC : ',scores,'\ncross_val_score :', round(np.mean(scores),4))

# ACC :  [0.37787591 0.43565315 0.45812891 0.50873391 0.45096354] 
# cross_val_score : 0.4463