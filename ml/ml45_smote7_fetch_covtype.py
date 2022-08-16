# 실습
# 증폭한 후 저장
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.datasets import fetch_covtype
import pickle

# 1. 데이터

dataset = fetch_covtype()

x = dataset.data
y = dataset.target

print(x.shape) # (581012, 54)
print(y.shape) # (581012,)
print(np.unique(y, return_counts=True)) # [1 2 3 4 5 6 7] [211840, 283301,  35754,   2747,   9493,  17367,  20510]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66, shuffle=True, stratify=y)


smote = SMOTE(random_state=66, k_neighbors=1)
x_train, y_train = smote.fit_resample(x_train, y_train)
print(x_train.shape) # (1586480, 54)
pickle.dump(x_train, open('D:\study_data\_save\_xg/covtype_smote_x_train.pkl', 'wb')) 
pickle.dump(y_train, open('D:\study_data\_save\_xg/covtype_smote_y_train.pkl', 'wb')) 