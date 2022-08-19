import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler, RobustScaler
from sklearn.preprocessing import QuantileTransformer, PowerTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
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

df =pd.DataFrame(x,y)
# print(df)
# print(df.head())


df.plot.box()
plt.title(class_name)
plt.xlabel('Feature')
plt.ylabel('데이터값')
# plt.show()

df['Ticket'] = np.log1p(df['Ticket'])
df['Name'] = np.log1p(df['Name'])

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
LogisticRegression 결과 : 0.8380
RandomForestClassifier 결과 : 0.8547
XGBClassifier 결과 : 0.8436

LogisticRegression 로그 변환 결과 : 0.8212
RandomForestClassifier 로그 변환 결과 : 0.8547
XGBClassifier 로그 변환 결과 : 0.8436    
 '''