import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import KFold,StratifiedKFold, cross_val_score, GridSearchCV, RandomizedSearchCV


#1.데이터
path = 'D:\study\_data\\'

datasets = pd.read_csv(path + 'winequality-white.csv', 
                   index_col=None, header=0, sep=';') #csv 파일은 통상 , or ; 형태로 되어 있음

# print(datasets.shape) #(4898, 12)
# print(datasets.head())
# print(datasets.describe())
# print(datasets.info())

datasets = datasets.values  # 판다스 넘파이 변환1
# datasets = datasets.to_numpy() # 판다스 넘파이 변환2
print(type(datasets))

x = datasets[:, :11]
y = datasets[:, 11]
print(x.shape, y.shape) #(4898, 11) (4898,)

def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out,
                                               [25,50,75]) 
                                               # 하위 25% 위치 값 Q1
                                               # 하위 50% 위치 값 Q2 (중앙값)
                                               # 하위 75% 위치 값 Q3
    print("1사분위 : ", quartile_1)
    print("q2 : ", q2)
    print("3사분위 : ", quartile_3)
    iqr = quartile_3 -quartile_1
    print("iqr : ", iqr) 
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    print('최소값 : ', lower_bound)
    print('최대값 : ', upper_bound)
    return np.where((data_out>upper_bound) |
                    (data_out<lower_bound))

outliers_loc = outliers(y)
print("이상치의 위치 :", outliers_loc)  
print('최소값 이하, 최대값 이상의 값을 찾아서 반환함 : ', outliers_loc)
print(len(outliers_loc[0])) # 200


print(np.unique(y, return_counts = True)) 
#  array([3., 4., 5., 6., 7., 8., 9.])
#  array([  20,  163, 1457, 2198,  880,  175,    5]                                              

x = np.delete(x, outliers_loc, 0) # outliers_loc의 위치에 있는 값을 삭제함
y = np.delete(y, outliers_loc, 0) # outliers_loc의 위치에 있는 값을 삭제함  

print(x.shape, y.shape)  #(4698, 11) (4698,)       
                                                         
'''

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
print("f1_score(macro) : ",f1_score(y_test, y_predict, average = 'macro'))
print("f1_score(micro) : ",f1_score(y_test, y_predict, average = 'micro'))

'''
