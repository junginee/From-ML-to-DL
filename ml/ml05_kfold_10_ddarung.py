import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split,KFold, cross_val_score 
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
import warnings
warnings.filterwarnings("ignore")

#1.데이터
path = './_data/ddarung/'
train_set = pd.read_csv(path + 'train.csv', index_col =0) 

print(train_set)
print(train_set.shape) #(1459,10)

test_set = pd.read_csv(path + 'test.csv', index_col =0) 
train_set = train_set.dropna() 
x = train_set.drop(['count'], axis = 1)
y = train_set['count'] #count 컬럼만 y 변수에 저장한다.

x_train,x_test,y_train, y_test = train_test_split(x,y, train_size=0.7, shuffle=True, random_state=72)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)
test_set = scaler.transform(test_set)

n_splits = 10
kfold = KFold(n_splits=n_splits, shuffle = True, random_state=300)

#2.모델구성
model = LinearSVR() 

#3,4. 컴파일, 훈련, 평가, 예측

# model.fit(x_train, y_train)
scores = cross_val_score(model,x,y,cv=kfold)
print('ACC : ',scores,'\ncross_val_score :', round(np.mean(scores),4))

# ACC :  [ 0.47131698 -0.13557164 -0.35824404 -0.13956935  0.47310749  0.45805179
#   0.04234422 -0.13172503  0.31290189  0.56269633]
# cross_val_score : 0.1555
