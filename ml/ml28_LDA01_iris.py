import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#1. 데이터
datasets =  load_iris()
x = datasets.data
y = datasets.target
print(x.shape) #(150, 4)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
scaler = StandardScaler()


x = scaler.fit_transform(x)
le = LabelEncoder()
y = le.fit_transform(y)

# LDA 
lda = LinearDiscriminantAnalysis(n_components=2)
lda.fit(x, y)
x = lda.transform(x)
print(x.shape) #(150, 2)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=72, shuffle=True, stratify=y)

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
# 걸린 시간 :  0.46370649337768555


#==================================== 결과 ==================================#
# XGBClassifier - gpu
# 결과 :  0.872421538170271      
# 걸린 시간 :  5.9461281299591064

# XGBClassifier - gpu / n_component : 10
# 결과 :  0.8419403974079843      
# 걸린 시간 :  4.058130979537964

# XGBClassifier - gpu / n_component : 20
# 결과 :  0.8866638554942643      
# 걸린 시간 :  4.79473614692688
#============================================================================#
