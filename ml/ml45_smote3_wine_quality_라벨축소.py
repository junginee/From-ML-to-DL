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
                   index_col=None, header=0, sep=';') 

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

for i in y:
    if 3 <= i <= 5:
        newlist += [0]
    elif 7 <= i <= 9:
        newlist += [2]
    else :
        newlist += [1] 
     
y = np.array(newlist)
print(np.unique(newlist, return_counts=True)) # array([1640, 2198, 1060]
     
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.8, 
                                                    random_state=123, shuffle = True, stratify = y)


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

print(pd.Series(y_train).value_counts()) 
# 1    1758
# 0    1312
# 2     848

print("================ SMOTE 적용 후 =================")
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=123, k_neighbors = 847) 
# Expected n_neighbors <= n_samples,  but n_samples = 848, n_neighbors = 1061
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


# model.score : 0.7286
# accuracy score :  0.7286
# f1_score(macro) :  0.7256
# f1_score(micro) :  0.7286

# ================ SMOTE 적용 후 =================       
# model.score : 0.7276
# accuracy score :  0.7276
# f1_score(macro) :  0.7253
# f1_score(micro) :  0.7276
