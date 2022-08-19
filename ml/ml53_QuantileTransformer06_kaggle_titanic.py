import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBClassifier, XGBRegressor 
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

#1. 데이터
path = './_data/kaggle_titanic/' # ".은 현재 폴더"
train_set = pd.read_csv(path + 'train.csv',
                        index_col=0)
test_set = pd.read_csv(path + 'test.csv', #예측에서 쓸거야!!
                       index_col=0)

# print(test_set) 
# print(train_set.isnull().sum()) 

train_set = train_set.fillna(train_set.median())

# print(test_set.isnull().sum())


drop_cols = ['Cabin']
train_set.drop(drop_cols, axis = 1, inplace =True)
test_set = test_set.fillna(test_set.mean())
train_set['Embarked'].fillna('S')
train_set = train_set.fillna(train_set.mean())

print("===============================")


print(train_set) 
print(train_set.isnull().sum())

test_set.drop(drop_cols, axis = 1, inplace =True)
cols = ['Name','Sex','Ticket','Embarked']

from tqdm import tqdm_notebook
for col in tqdm_notebook(cols):
    le = LabelEncoder()
    train_set[col]=le.fit_transform(train_set[col])
    test_set[col]=le.fit_transform(test_set[col])
    
x = train_set.drop(['Survived'],axis=1) #axis는 컬럼 
print(x) #(891, 9)

y = train_set['Survived']
print(y.shape) #(891,)

gender_submission = pd.read_csv(path + 'gender_submission.csv',#예측에서 쓸거야!!
                       index_col=0)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, test_size = 0.2, random_state=123,
)


scaler = [StandardScaler(), MinMaxScaler(), MaxAbsScaler(), RobustScaler(), 
QuantileTransformer(),PowerTransformer(method='yeo-johnson'),] #PowerTransformer(method='box-cox')]

model = RandomForestClassifier()

for x in scaler :
  scaler = x         
  x_train = scaler.fit_transform(x_train)
  x_test = scaler.transform(x_test)
  model.fit(x_train, y_train)
  y_predict = model.predict(x_test)
  results = accuracy_score(y_test, y_predict)
  score = f1_score(y_test, y_predict, average = 'macro')
  class_name = scaler.__class__.__name__
  print('{0} 결과 : {1:.4f}'.format(scaler,results))
  print('{0} f1_score결과 : {1:.4f}'.format(scaler,score))
  print() 
  
'''
StandardScaler() 결과 : 0.8715
StandardScaler() f1_score결과 : 0.8632

MinMaxScaler() 결과 : 0.8827
MinMaxScaler() f1_score결과 : 0.8728

MaxAbsScaler() 결과 : 0.8659
MaxAbsScaler() f1_score결과 : 0.8551

RobustScaler() 결과 : 0.8883
RobustScaler() f1_score결과 : 0.8792

QuantileTransformer() 결과 : 0.8715
QuantileTransformer() f1_score결과 : 0.8616

PowerTransformer() 결과 : 0.8771
PowerTransformer() f1_score결과 : 0.8680
'''  