import numpy as np
import pandas as pd
from keras.datasets import mnist
from sklearn.decomposition import PCA
from tensorflow.python.keras. models import Sequential, load_model
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier
from tensorflow.python.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split



#1. 데이터
import time
start = time.time() # 시작 시간 체크
(x_train, y_train), (x_test, y_test) = mnist.load_data() # (60000, 28, 28) (10000, 28, 28)
x = np.append(x_train, x_test, axis=0) # (70000, 28, 28)
x = x.reshape(x.shape[0], x.shape[1]*x.shape[2]) # (70000, 784)
print(x.shape) # (70000, 784)
y= np.append(y_train, y_test) # (70000,)


pca = PCA(n_components=712)
x = pca.fit_transform(x)
pca_EVR = pca.explained_variance_ratio_ 
cumsum = np.cumsum(pca_EVR)

# print('n_components=', 783, ':') 
# print(np.argmax(cumsum >= 0.95)+1) 
# print(np.argmax(cumsum >= 0.99)+1) 
# print(np.argmax(cumsum >= 0.999)+1) 
# print(np.argmax(cumsum+1)) #712
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=66)


# model = RandomForestClassifier()
model = XGBClassifier()

model.fit(x_train, y_train)

result = model.score(x_test, y_test)
end = time.time() # 종료 시간 체크

print('실행 시간 :', end-start)
print('accuracy :', result)


# 1. 나의 최고의 DNN
# time = 87.61148142814636
# acc = 0.9375
# keras30_dnn1_mnist


# 2. 나의 최고의 CNN
# time = 928.7224872112274
# acc = 0.8612
# keras29_Fashion_mnist_CNN


# 3. PCA 0.95  #154
# time = ???
# acc = ???


# 4. PCA 0.99  #331
# time = ???
# acc = ???


# 5. PCA 0.999  #486
# time = ???
# acc = ???


# 5. PCA 1.0  #713 
# time =  1094.127665758133
# acc =  0.9574285714285714

# 1. 나의 최고의 DNN
# 시간:  362.1198184490204
# acc스코어 :  0.9445

# 2. 나의 최고의 CNN
# 시간:  152.1847949028015
# acc스코어 :  0.9765

# 3. PCA 0.95
# 154 의 결과:  0.9649
# 시간:  9.673815965652466

# 4. PCA 0.99
# 331 의 결과:  0.9631
# 시간:  14.473661184310913

# 5. PCA 0.999
# 486 의 결과:  0.963
# 시간:  18.711859703063965

# 6. PCA 1.0
# 713 의 결과:  0.9633
# 시간:  24.653253316879272
