from pyexpat import model
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor 
import time


#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, random_state=123, train_size=0.8, stratify=y
)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = StratifiedKFold(n_splits=n_splits, shuffle = True, random_state=123)


parameters = {'n_estimators' : [100], 
              'learning_rate' : [0.1],
              'max_dapth' : [None],
              'gamma' : [0],
              'min_child_weight' :[0.5],
              'subsample' : [0.5],
              'colsample_bytree' : [0.3],
              'colsample_bylevel' : [1],
              'colsample_bynode' : [1],
              'reg_alpha' : [0],
              'reg_labda' : [0]}



#2. 모델
model = XGBClassifier(random_state = 123,
                    n_estimators = 1000,
                    learning_rate = 0.1,
                    max_dapth=3,
                    gamma=1)

#3. 훈련
start = time.time()
model.fit(x_train, y_train, verbose = 1, early_stopping_rounds=200,
                eval_set = [ (x_train, y_train), (x_test, y_test) ],    
                eval_metric='error')
                                         
                        
                # eval_metric 종류
                # 회귀 : rmse, mae, rmsle..
                # 이진 : error, auc, logloss
                # 다중 : merror, mlogloss..          


# - early_stopping_rounds: 더 이상 비용 평가 지표가 감소하지 않는 최대 반복횟수
# - eval_metric: 반복 수행 시 사용하는 비용 평가 지표
# - eval_set: 평가를 수행하는 별도의 검증 데이터 세트. 일반적으로 검증 데이터 세트에서 반복적으로 비용감소 성능 평가

end = time.time()
print("걸린시간 : ", end-start)


#4. 평가

results = model.score(x_test, y_test)
print('점수 :', results)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print("최종 test 점수 :", acc)

# 걸린시간 :  0.3981049060821533
# 점수 : 0.9736842105263158   
# 최종 test 점수 : 0.9736842105263158
