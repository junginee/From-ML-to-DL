import numpy as np
import pandas as pd

from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

#1. 데이터
datasets = load_iris()

# df =pd.DataFrame(datasets.data, columns=datasets.feature_names) #데이터프레임 형태로 전환
# print(df.head(7))


x_train, x_test, y_train, y_test = train_test_split(
    datasets.data, datasets.target, train_size=0.8, random_state=123,
    stratify=datasets.target
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

xg = XGBClassifier(learning_rate=0.01,
              reg_alpha=0.01,
              reg_lambd=1,)  

lg = LGBMClassifier()
cat = CatBoostClassifier()

model = VotingClassifier(
    estimators=[('xg',xg),('lg',lg), ('cb',cat)],
    voting = 'soft'      #hard
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
    
# XGBClassifier 정확도 : 0.9667
# LGBMClassifier 정확도 : 0.9667
# CatBoostClassifier 정확도 : 0.9667    