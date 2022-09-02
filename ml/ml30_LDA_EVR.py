import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
import xgboost as xg
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

#1. 데이터
datasets =  load_digits() #(1797, 64)->(1797, 9)

x = datasets.data
y = datasets.target
print(x.shape) 

lda = LinearDiscriminantAnalysis()
lda.fit(x, y)
x = lda.transform(x)
print(x.shape)

lda_EVR = lda.explained_variance_ratio_

cumsum = np.cumsum(lda_EVR)
print(cumsum)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=72, shuffle=True, stratify=y)

print(x_train.shape, x_test.shape)  
print(np.unique(y_train, return_counts=True))   

'''
#2. 모델
model = XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0)

#3. 훈련
import time
start = time.time()
model.fit(x_train, y_train)
end = time.time()

#4. 평가, 예측
result = model.score(x_test, y_test)
print('결과 : ', result)
print('걸린 시간 : ', end - start)
'''
