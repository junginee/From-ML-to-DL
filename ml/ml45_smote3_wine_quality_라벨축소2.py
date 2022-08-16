import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import KFold,StratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV


#1.데이터
path = 'D:\study\_data\\'

datasets = pd.read_csv(path + 'winequality-white.csv', 
                   index_col=None, header=0, sep=';') #csv 파일은 통상 , or ; 형태로 되어 있음

print(datasets.shape) #(4898, 12)
print(datasets.head())
print(datasets.describe())
print(datasets.info())


print(type(datasets))  
# print(datasets2.shape)

x = datasets.to_numpy()[:, :11]
y = datasets.to_numpy()[:, 11]
print(x.shape, y.shape)

print(np.unique(y, return_counts = True))    
print(datasets['quality'].value_counts())
                         
print(y[:20])

newlist =[]

for index,value in enumerate(y):
    if value == 9 :  # 9.0       5
     y[index] = 7
    elif value == 8 : # 8.0     175
       y[index] = 7
    elif value == 7 : # 7.0     880
       y[index] = 7
    elif value == 6 : # 6.0     2198
       y[index] = 6
    elif value == 5 : # 5.0     1457
       y[index] = 5
    elif value == 4 : # 4.0      103
       y[index] = 4
    elif value == 3 : # 3.0      20
       y[index] = 4
    else :
       y[index] = 0
       
   
      
    
print('[newlist] \n', np.unique(newlist, return_counts=True)) 
# array([], dtype=float64), array([], dtype=int64))
     
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.8, 
                                                    random_state=123, shuffle = True, stratify = y)

print(pd.Series(y_train).value_counts()) 
# 6.0    1758
# 5.0    1166
# 7.0     848
# 4.0     146


#2. 모델
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier( )

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
y_predict = model.predict(x_test)

score = model.score(x_test, y_test)

from sklearn.metrics import accuracy_score, f1_score
print('model.score :' , round(score,4))
print("accuracy score : ",
      round(accuracy_score(y_test, y_predict), 4 ))
print("f1_score(macro) : ",round(f1_score(y_test, y_predict, average = 'macro'),4))
print("f1_score(micro) : ",round(f1_score(y_test, y_predict, average = 'micro'),4))

print("================ SMOTE 적용 후 =================")
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=123, k_neighbors = 144
              ) 

smote.fit_resample(x_train, y_train)  #test 데이터는 예측하기 위해 smote 적용하지 않는다.

model = RandomForestClassifier()
model.fit(x_train, y_train)               

y_predict = model.predict(x_test)

score = model.score(x_test, y_test)

from sklearn.metrics import accuracy_score, f1_score
print('model.score :' , round(score,4))
print("accuracy score : ",
      round(accuracy_score(y_test, y_predict), 4 ))
print("f1_score(macro) : ",round(f1_score(y_test, y_predict, average = 'macro'),4))
print("f1_score(micro) : ",round(f1_score(y_test, y_predict, average = 'micro'),4))

# model.score : 0.7235
# accuracy score :  0.7235
# f1_score(macro) :  0.6372
# f1_score(micro) :  0.7235
# ================ SMOTE 적용 후 =================       
# model.score : 0.7357
# accuracy score :  0.7357
# f1_score(macro) :  0.6374
# f1_score(micro) :  0.7357