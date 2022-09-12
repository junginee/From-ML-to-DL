import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings("ignore")

#1.데이터

path = './_data/kaggle_titanic/' 
train_set = pd.read_csv(path + 'train.csv',
                        index_col=0)
test_set = pd.read_csv(path + 'test.csv', 
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
    
x = train_set.drop(['Survived'],axis=1) 
print(x) #(891, 9)

y = train_set['Survived']
print(y.shape) #(891,)

gender_submission = pd.read_csv(path + 'gender_submission.csv',
                       index_col=0)




x_train, x_test, y_train, y_test = train_test_split(
    x,y, train_size=0.8,
    shuffle=True ,random_state=100,stratify=y)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

xg = XGBClassifier(learning_rate=0.2,
              reg_alpha=0.02,
              min_child_weight=1.5,
             )  

lg = LGBMClassifier()
cat = CatBoostClassifier()

model = VotingClassifier(
    estimators=[('xg',xg),('lg',lg), ('cb',cat)],
    voting = 'soft'      
)

#3. 훈련
model.fit(x_train, y_train)


#4. 평가, 예측
y_predict = model.predict(x_test)

score = accuracy_score(y_test, y_predict)
print('score :', round(score,4))


classfiers = [xg,lg,cat]
for model2 in classfiers:
    model2.fit(x_train, y_train, verbose = 0)
    y_predict = model2.predict(x_test)
    score2 = accuracy_score(y_test, y_predict)
    class_name = model2.__class__.__name__
    print('{0} 정확도 : {1:.4f}'.format(class_name,score))
    
# score : 0.8827
# XGBClassifier 정확도 : 0.8827
# LGBMClassifier 정확도 : 0.8827
# CatBoostClassifier 정확도 : 0.8827
