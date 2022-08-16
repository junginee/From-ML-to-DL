# 왜 정확도가 낮을까?
# 데이터 분포가 골고루게 분포되어 있지 않기 때문이다.
# 분류에서는 y 값 분포 확인하기

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

#  fixed acidity  volatile acidity  citric acid  residual sugar  chlorides  free sulfur dioxide  total sulfur dioxide  density    pH  sulphates  alcohol  quality
# 0            7.0              0.27         0.36            20.7      0.045                 45.0                 170.0   1.0010  3.00       0.45      8.8        6
# 1            6.3              0.30         0.34             1.6      0.049                 14.0                 132.0   0.9940  3.30       0.49      9.5        6
# 2            8.1              0.28         0.40             6.9      0.050                 30.0                  97.0   0.9951  3.26       0.44     10.1        6
# 3            7.2              0.23         0.32             8.5      0.058                 47.0                 186.0   0.9956  3.19       0.40      9.9        6
# 4            7.2              0.23         0.32             8.5      0.058                 47.0                 186.0   0.9956  3.19       0.40      9.9        6

#  #   Column                Non-Null Count  Dtype
# ---  ------                --------------  -----
#  0   fixed acidity         4898 non-null   float64
#  1   volatile acidity      4898 non-null   float64
#  2   citric acid           4898 non-null   float64
#  3   residual sugar        4898 non-null   float64
#  4   chlorides             4898 non-null   float64
#  5   free sulfur dioxide   4898 non-null   float64
#  6   total sulfur dioxide  4898 non-null   float64
#  7   density               4898 non-null   float64
#  8   pH                    4898 non-null   float64
#  9   sulphates             4898 non-null   float64
#  10  alcohol               4898 non-null   float64
#  11  quality               4898 non-null   int64
#                                            => 분류모델에서는 y 값이 int 형식이어야 한다.
#                                            => 만약 float 형식이라면 int로 바꿀지 고민해볼 필요가 있다.

datasets = datasets.values  # 판다스 넘파이 변환1
# datasets = datasets.to_numpy() # 판다스 넘파이 변환2
print(type(datasets))  

x = datasets[:, :11]
y = datasets[:, 11]

print(x.shape, y.shape)
print(np.unique(y, return_counts = True)) # 다중분류에서는 y 데이터 분포 반드시 확인할 것!
                                          # 1. 넘파이
                                          # np.unique(y, return_counts = True를 통해
                                          # 2. 판다스
                                          # print(datasets['quality'].value_counts())          
                                          
                                        #  array([3., 4., 5., 6., 7., 8., 9.])
                                        #  array([  20,  163, 1457, 2198,  880,  175,    5]                                

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.8, 
                                                    random_state=123, shuffle = True, stratify = y)

# [XG boost 모델 사용 시 label encoder]
# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# y_train = le.fit_transform(y_train)
# y_test = le.transform(y_test)

# le.classes_

'''
# Invalid classes inferred from unique values of `y`.  Expected: [0 1 2 3 4 5 6], got [3 4 5 6 7 8 9]
# xg_boost 모델 사용 시 이런 에러가 나타나면?
# y 값에 대해 labelencoder 처리를 해줘야한다.
# 처리하게 되면 [3 4 5 6 7 8 9] --> [0 1 2 3 4 5 6]로 변환되며 모델이 돌아가게 됨
# (y 값이 int 형식일지라도 labelencoder 처리를 해줘야 한다.)
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
print("f1_score(macro) : ",f1_score(y_test, y_predict, average = 'macro'))
print("f1_score(micro) : ",f1_score(y_test, y_predict, average = 'micro'))

#다중분류에서는 f1_score를 사용할 수 없다. 
#so, 컬럼별로 f1_score를 구한 후 이 값들의 평균을 구한다.



