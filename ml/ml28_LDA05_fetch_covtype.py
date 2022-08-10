import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.datasets import load_wine, fetch_covtype
from sklearn.datasets import load_digits
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#1. 데이터
datasets =  fetch_covtype()
x = datasets.data
y = datasets.target
print(x.shape) #(581012, 54)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
# scaler = MaxAbsScaler()
# scaler = RobustScaler()

x = scaler.fit_transform(x)
le = LabelEncoder()
y = le.fit_transform(y)

print(np.unique(y))  #[0 1 2 3 4 5 6]

# pca = PCA(n_components=20)  
# x = pca.fit_transform(x)
# print(x.shape)  

# pca_EVR = pca.explained_variance_ratio_
# cumsum = np.cumsum(pca_EVR) 
# print(cumsum)

# LDA (LDA의 n_components에 들어가는 값은 'y라벨-1' 이하 값 기재)
lda = LinearDiscriminantAnalysis(n_components=6)
lda.fit(x, y)
x = lda.transform(x)
print(x.shape) 

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=72, shuffle=True, stratify=y)

# LDA 
# lda = LinearDiscriminantAnalysis(n_components=6)
# lda.fit(x_train, y_train)
# x_train = lda.transform(x_train)
# x_test = lda.transform(x_test)
print(x_train.shape, x_test.shape) 
print(np.unique(y_train, return_counts=True))  
#(array([0, 1], dtype=int64), array([170, 285], dtype=int64))


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


# LDA 결과((n_components=6)
# 결과 :  0.7891018304174591
# 걸린 시간 :  3.571747064590454

# LDA 결과((n_components=5)
# 결과 :  0.7739731332237549
# 걸린 시간 :  3.3884549140930176

# LDA 결과((n_components=4)
# 결과 :  0.7599889847938521
# 걸린 시간 :  3.261166572570801

# LDA 결과((n_components=3)
# 결과 :  0.7475538497284924
# 걸린 시간 :  3.0792860984802246
