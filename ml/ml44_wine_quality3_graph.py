# 아웃라이어 확인
# 아웃라이어 처리

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

print(datasets.shape) #(4898, 12)
print(datasets.head())
print(datasets.describe())
print(datasets.info())

################## 그래프 그리기 ################
#1. value_counts 쓰지말것
#2. groupby, count 사용
# plt.bar로 그린다. (quality 컬럼)

import matplotlib.pyplot as plt

count_data = datasets.groupby('quality')['quality'].count()
print(count_data)

plt.bar(count_data.index, count_data)
plt.show() 
