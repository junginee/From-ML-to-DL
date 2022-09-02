import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#1. 데이터
datasets =  load_wine()
x = datasets.data
y = datasets.target
print(x.shape) #(178, 13)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

x = scaler.fit_transform(x)
le = LabelEncoder()
y = le.fit_transform(y)

print(np.unique(y))  #[0 1 2]

# LDA (LDA의 n_components는 y라벨 -1 이하)
lda = LinearDiscriminantAnalysis(n_components=2)
lda.fit(x, y)
x = lda.transform(x)
print(x.shape) #(150, 2)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=72, shuffle=True, stratify=y)

print(x_train.shape, x_test.shape)  # (120, 2) (30, 2)
print(np.unique(y_train, return_counts=True))   
#(array([0, 1, 2], dtype=int64), array([40, 40, 40], dtype=int64))

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


# LDA 결과
# 결과 :  1.0
# 걸린 시간 :  0.4827096462249756
