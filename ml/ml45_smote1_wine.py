from winsound import SND_PURGE
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import sklearn as sk
print( '사이킷런 :', sk.__version__)

#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets['target']
print(x.shape, y.shape)
print(type(x))
print(np.unique(y, return_counts = True))
print(pd.Series(y).value_counts())

# 1    71
# 0    59
# 2    48

# x = x[ :-25]
# y = y[ :-25]
# print(pd.Series(y_new).value_counts())
print(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.75, shuffle= True, random_state=123,
    stratify=y)

'''
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

# 기본 결과
# accuracy score :  0.9778 
# f1_score(macro) :  0.9797                        
#                                                   
# 데이터 축소 후(2 라벨을 30개 축소 후)             
# accuracy score :  0.9429                          
# f1_score(macro) :  0.8596    ==> 축소 이유 무엇일까

'''

print(pd.Series(y_train).value_counts()) 
print("================ SMOTE 적용 후 =================")
smote = SMOTE(random_state=123, k_neighbors=16)
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