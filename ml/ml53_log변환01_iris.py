import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from xgboost import XGBClassifier, XGBRegressor 
import warnings
warnings.filterwarnings("ignore")

#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

print(x.shape, y.shape) #(506, 13) (506,)


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
# print(df)
# print(df.head())

df.plot.box()
plt.title(class_name)
plt.xlabel('Feature')
plt.ylabel('데이터값')
plt.show()


df['sepal length (cm)'] = np.log1p(df['sepal length (cm)']) # LinearRegression 로그 변환 결과 : 0.7596
# df['ZN'] = np.log1p(df['ZN']) # LinearRegression 로그 변환 결과 : 0.7734
# df['TAX'] = np.log1p(df['TAX']) # LinearRegression 로그 변환 결과 : 0.7669
                                # LinearRegression 3개 모두 로그 변환 결과 : 0.7667

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

'''
LogisticRegression 결과 : 1.0000
RandomForestClassifier 결과 : 1.0000
XGBClassifier 결과 : 1.0000

LogisticRegression 로그 변환 결과 : 1.0000
RandomForestClassifier 로그 변환 결과 : 1.0000
XGBClassifier 로그 변환 결과 : 1.0000
'''
