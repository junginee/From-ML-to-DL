import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR

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


#2. 모델구성
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression, LinearRegression  #LogisicRegression 분류
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn. ensemble import RandomForestClassifier, RandomForestRegressor

model = Perceptron(),LinearSVC(),LinearRegression(),KNeighborsRegressor(),DecisionTreeRegressor(),RandomForestRegressor()


for i in model:    
    model = i
    

    #3. 컴파일, 훈련
    model.fit(x_train, y_train)


    #4. 평가, 예측

    result = model.score(x_test,y_test)   

    y_predict = model.predict(x_test)

    print(f"{i} : ", round(result,4))
    
# Perceptron() :  0.0125
# LinearSVC() :  0.0251
# LinearRegression() :  0.5859
# KNeighborsRegressor() :  0.6746
# DecisionTreeRegressor() :  0.6389
# RandomForestRegressor() :  0.7583   