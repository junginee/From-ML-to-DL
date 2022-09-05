from sklearn.datasets import  load_diabetes
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor 
import time


#1. 데이터
datasets =load_diabetes()
x = datasets.data
y = datasets.target

print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, random_state=123, train_size=0.8)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle = True, random_state=123)

parameters = {'n_estimators' : [500], 
              'learning_rate' : [0.1],
              'max_dapth' : [None],
              'gamma' : [100],
              'min_child_weight' :[100],
              'subsample' :  [0, 0.1, 0.2, 0.3, 0.5, 0.7, 1],
            #   'colsample_bytree' : [0.3],
            #   'colsample_bylevel' : [1],
            #   'colsample_bynode' : [1],
            #   'reg_alpha' : [0],
               'reg_lamda' : [0, 0.1, 0.01, 0.001, 1, 2, 10]
            }



#2. 모델
xgb = XGBRegressor(random_state = 123,
                    )

model = GridSearchCV(xgb, parameters, cv= kfold, n_jobs=8)

#3. 훈련
model.fit(x_train, y_train)

print("최상의 매개변수 :", model.best_params_)
print("최상의 점수 : ", model.best_score_)

#4. 평가
y_predict = model.predict(x_test)
print('r2 score :', round(r2_score(y_test, y_predict),4))

###############################################
# r2 score : 0.5794
###############################################


# 최상의 매개변수 : {'learning_rate': 0.01, 'max_dapth': None, 'n_estimators': 500}
# 최상의 점수 :  0.26387608276474595      
# r2 score : 0.4935 


# 최상의 매개변수 : {'gamma': 100, 'learning_rate': 0.1, 'max_dapth': None, 'n_estimators': 500}
# 최상의 점수 :  0.23510702062268204      
# r2 score : 0.4656


# 최상의 매개변수 : {'gamma': 100, 'learning_rate': 0.1, 'max_dapth': None, 'min_child_weight': 100, 
#                        'n_estimators': 500} 
# 최상의 점수 :  0.38031524916289766      
# r2 score : 0.5794


# 최상의 매개변수 : {'gamma': 100, 'learning_rate': 0.1, 'max_dapth': None, 
#              'min_child_weight': 100, 'n_estimators': 500, 
#              'reg_lamda': 0, 'subsample': 1}
# 최상의 점수 :  0.38031524916289766      
# r2 score : 0.5794
