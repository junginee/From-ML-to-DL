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
train_set = pd.read_csv(path + 'train.csv', index_col =0) #id는 0번째에 위치한다. #[1459 rows x 10 columns]

print(train_set)
print(train_set.shape) #(1459,10)

test_set = pd.read_csv(path + 'test.csv', index_col =0) 
print(test_set)
print(test_set.shape)  #(715, 9)

print(train_set.columns)
print(train_set.info())  
print(train_set.describe())


print(train_set.isnull().sum()) #train set에 있는 널값의 합계를 구한다.
train_set = train_set.dropna() #결측치가 들어있는 행을 삭제한다.
print(train_set.isnull().sum()) #결측치 제거 후 train set에 들어있는 널값의 합계를 구한다.
x = train_set.drop(['count'], axis = 1) #x 변수에는 count 열을 제외한 나머지 컬럼을 저장한다.

print(x)
print(x.columns)
print(x.shape) #(1459,9)

y = train_set['count'] #count 컬럼만 y 변수에 저장한다.
print(y)
print(y.shape)

x_train,x_test,y_train, y_test = train_test_split(x,y, train_size=0.7, shuffle=True, random_state=72)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)
test_set = scaler.transform(test_set)
print(np.min(x_train))  # 0.0
print(np.max(x_train))  # 1.0

print(np.min(x_test))  # 1.0
print(np.max(x_test))  # 1.0

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
