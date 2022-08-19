import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor 
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

#1. 데이터
path = 'D:\study\_data\ddarung\\'
train_set = pd.read_csv(path+'train.csv',index_col=0) # index_col = n : n번째 칼럼을 인덱스로 인식
# print(train_set)
# print(train_set.shape) # (1459, 10)

test_set = pd.read_csv(path+'test.csv', index_col=0)
# print(test_set)
# print(test_set.shape) # (715, 9)

# 결측치 중간값으로
print(train_set.info())
print(train_set.isnull().sum()) # 결측치 전부 더함
median = train_set.median()
train_set = train_set.fillna(median) # 결측치 중간값으로
print(train_set.isnull().sum()) # 없어졌는지 재확인

x = train_set.drop(['count'], axis=1) # axis = 0은 열방향으로 쭉 한줄(가로로 쭉), 1은 행방향으로 쭉 한줄(세로로 쭉)
y = train_set['count']


x_train, x_test, y_train, y_test = train_test_split(
    x,y, test_size = 0.2, random_state=1234,
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2.모델
model = LinearRegression(), RandomForestRegressor(), XGBRegressor()


for i in model :
    model = i
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    results = r2_score(y_test, y_predict)
    class_name = model.__class__.__name__
    print('{0} 결과 : {1:.4f}'.format(class_name,results))
print()    

#################### 로그 변환 ###################

df =pd.DataFrame(x, y)
print(df) 
print(df.head())

df.plot.box()
plt.title('ddarung')
plt.xlabel('Feature')
plt.ylabel('데이터값')
plt.show()

# print("로그변환 전 :",df['B'].head())
# df['B'] = np.log1p(df['B'])
# print("로그변환 후 :",df['B'].head())

df['hour_bef_visibility '] = np.log1p(df['hour_bef_visibility ']) 
# LinearRegression 로그 변환 결과 : 0.6074
# RandomForestRegressor 로그 변환 결과 : 0.8046
# XGBRegressor 로그 변환 결과 : 0.8266

# df['Population'] = np.log1p(df['Population']) 
# LinearRegression 로그 변환 결과 : 0.6066
# RandomForestRegressor 로그 변환 결과 : 0.8039
# XGBRegressor 로그 변환 결과 : 0.8267


x_train, x_test, y_train, y_test = train_test_split(
    x,y, test_size = 0.2, random_state=1234,
)

# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

#2.모델
model = LinearRegression(), RandomForestRegressor(), XGBRegressor()

for i in model :
    model = i
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    results = r2_score(y_test, y_predict)
    class_name = model.__class__.__name__
    print('{0} 로그 변환 결과 : {1:.4f}'.format(class_name,results))
