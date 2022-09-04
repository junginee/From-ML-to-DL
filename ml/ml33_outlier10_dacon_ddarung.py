import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split,KFold, cross_val_score,GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from sklearn.utils import all_estimators
from sklearn.metrics import r2_score
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

print(train_set.isnull().sum())
train_set = train_set.dropna() 

print(train_set.isnull().sum()) 
x = train_set.drop(['count'], axis = 1)

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

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle = True, random_state=66)


parameters = [
        {'n_estimators':[100,200], 'max_depth':[6,8,10,12], 'min_samples_split':[2,3,5,10]},
        {'n_estimators':[100,200], 'min_samples_leaf':[3,5,7,10], 'min_samples_split':[2,3,5,10]},
]                                                                       


#2. 모델구성
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron, LogisticRegression # LogisticRegression는 분류임
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor

# model =  RandomForestClassifier(max_depth=10, min_samples_split=3)                         
model = RandomizedSearchCV(RandomForestRegressor(), parameters, cv=kfold, verbose=1,          
                     refit=True, n_jobs=1)                            


#3. 컴파일, 훈련
model.fit(x_train, y_train)

print("최적의 매개변수 : ", model.best_estimator_)
# 최적의 매개변수 :  RandomForestRegressor(max_depth=10)
print("최적의 파라미터 : ", model.best_params_)
#최적의 파라미터 :  {'n_estimators': 100, 'min_samples_split': 2, 'max_depth': 10}
print("best_score_ : ", model.best_score_)
# best_score_ :  0.7519372504373839   
print("model.score : ", model.score(x_test, y_test))
# model.score :  0.7584977389365182  

#4. 평가
y_predict = model.predict(x_test)
print("r2_score", round(r2_score(y_test, y_predict),4))


# 결측치 처리 방식
# 삭제 > 평균 > 중위값

# r2_score 0.7585 (삭제)
# 결측치 삭제 후 훈련시켰을 때의 값이 가장 좋으며 
# 그 결과값은 이전과 크게 차이나지 않음
