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

datasets = datasets.values
# datasets = datasets.to_numpy() # 판다스 넘파이 변환2
print(type(datasets))  


x = datasets[:, :11]
y = datasets[:, 11]

# x = datasets[:-10]
# y = datasets[:-10]

print(x.shape, y.shape)
print(np.unique(y, return_counts = True)) #(4873, 12) (4873, 12)                               
print(pd.Series(y).value_counts())

x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size = 0.8, random_state=123, shuffle = True, 
    stratify = y)
print(pd.Series(y_train).value_counts()) 
#y_train 확인한 이유? smote 적용 시 k_neighbors 값 설정을 위해서
# 6.0    1758
# 5.0    1166
# 7.0     704
# 8.0     140
# 4.0     130
# 3.0      16
# 9.0       4
# k_neighbors는 4 미만이어야 한다. 

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



print("================ SMOTE 적용 후 =================")
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=123, k_neighbors = 3
              ) #k_neighbors default = 5
# Expected n_neighbors <= n_samples,  but n_samples = 4, n_neighbors = 5
# 

smote.fit_resample(x_train, y_train)

model = RandomForestClassifier()
model.fit(x_train, y_train)               

y_predict = model.predict(x_test)

score = model.score(x_test, y_test)

from sklearn.metrics import accuracy_score, f1_score
print('model.score :' , round(score,4))
print("accuracy score : ",
      round(accuracy_score(y_test, y_predict), 4 ))
print("f1_score(macro) : ",round(f1_score(y_test, y_predict, average = 'macro'),4))
print("f1_score(micro) : ",round(f1_score(y_test, y_predict, average = 'micro'),4)
