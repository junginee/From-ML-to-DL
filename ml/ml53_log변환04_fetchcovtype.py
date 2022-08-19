import numpy as np
import pandas as pd
from sklearn.datasets import  fetch_covtype 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from xgboost import XGBClassifier, XGBRegressor 
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

#1. 데이터
datasets =  fetch_covtype ()
x = datasets.data
y = datasets.target

le = LabelEncoder()
y = le.fit_transform(y)

print(x.shape, y.shape) #(506, 13) (506,)

df =pd.DataFrame(datasets.data, columns=[datasets.feature_names])
print(df)


x_train, x_test, y_train, y_test = train_test_split(
    x,y, test_size = 0.2, random_state=1234,
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2.모델
model = LogisticRegression(), RandomForestClassifier(), XGBClassifier()


for i in model :
    model = i
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    results = accuracy_score(y_test, y_predict)
    class_name = model.__class__.__name__
    print('{0} 결과 : {1:.4f}'.format(class_name,results))
print()    

#################### 로그 변환 ###################

df =pd.DataFrame(datasets.data, columns=[datasets.feature_names])
print(df)
# print(df.head())

df.plot.box()
plt.title(class_name)
plt.xlabel('Feature')
plt.ylabel('데이터값')
# plt.show()

df['Elevation '] = np.log1p(df['Elevation ']) # LinearRegression 로그 변환 결과 : 0.7596

x_train, x_test, y_train, y_test = train_test_split(
    x,y, test_size = 0.2, random_state=1234,
)

# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

#2.모델
model = LogisticRegression(), RandomForestClassifier(), XGBClassifier()

for i in model :
    model = i
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    results = accuracy_score(y_test, y_predict)
    class_name = model.__class__.__name__
    print('{0} 로그 변환 결과 : {1:.4f}'.format(class_name,results))
  