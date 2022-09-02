import numpy as np    
from sklearn.model_selection import train_test_split 
from sklearn.decomposition import PCA
from keras.datasets import mnist 
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
import time

(x_train, y_train), (x_test, y_test)=mnist.load_data()  

# print(y_train.shape)
x_train = x_train.reshape(60000,784)
x_test = x_test.reshape(10000,784)

scaler = MinMaxScaler()
scaler.fit(x_train) #여기까지는 스케일링 작업을 했다.
scaler.transform(x_train)
x_train_scale = scaler.transform(x_train)
x_test_scale = scaler.transform(x_test)

lda = LinearDiscriminantAnalysis(n_components=2)
lda.fit(x_train, y_train)
x_train = lda.transform(x_train)
print(x_train.shape) #(150, 2)

from sklearn.model_selection import KFold,cross_val_score,GridSearchCV,StratifiedKFold, RandomizedSearchCV
kfold = StratifiedKFold(n_splits=5,shuffle=True,random_state=100)

parameters = [
    {"n_estimators":[100,200,300],"learning_rate":[0.1, 0.3, 0.001,0.01],
     "max_depth": [4,5,6]},
    # {"n_estimators":[90,100,110],"learning_rate":[0.1, 0.001, 0.01],
    #  "max_depth": [4,5,6]},
    {"n_estimators":[90,110],"learning_rate":[0.1, 0.001, 0.5],
     "max_depth": [4,5,6],"colsample_bytree":[0.6,0.9,1],
     "colsample_bylevel":[0.6,0.7,0.9]}
]

model = RandomizedSearchCV(XGBClassifier(tree_method='gpu_hist'),parameters,cv=kfold,verbose=2,
                     refit=True,n_jobs=-1) 


import time
start = time.time()
model.fit(x_train, y_train)
end = time.time()


result = model.score(x_test, y_test)
print('결과 : ', result)
print('걸린 시간 : ', end - start)
