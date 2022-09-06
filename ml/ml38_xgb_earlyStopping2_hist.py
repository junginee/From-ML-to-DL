import numpy as np
from sklearn.datasets import load_breast_cancer, load_boston
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, r2_score
import time

# 1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, random_state=1234, train_size=0.8, stratify=y)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 2. 모델
model = XGBClassifier(n_estimators=1000,
              learning_rate=1,
              max_depth=2,
              gamma=0,
              min_child_weight=1,
              subsample=1,
              colsample_bytree=0.5,
              colsample_bylevel=1,
              colsample_bynode=1,
              reg_alpha=0.01,
              reg_lambd=1,
              tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0, random_state=1234
              )

model.fit(x_train, y_train, early_stopping_rounds=10, 
          eval_set=[(x_train, y_train), (x_test, y_test)],
          eval_metric=['logloss'])


#4. 평가
results = model.score(x_test, y_test)
print('점수 :', results)

y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print("최종 test 점수 :", acc)

print("===================================")
hist = model.evals_result()
print(hist)

import matplotlib.pyplot as plt
plt.figure(figsize=(9,6))
plt.plot( hist['validation_0']['logloss'], marker=".", c='red', alpha=0.3, label='train_set' )
plt.plot( hist['validation_1']['logloss'], marker='.', c='blue', alpha =0.3,label='test_set' )
plt.grid() 
plt.title('loss_error')
plt.ylabel('loss_error')
plt.xlabel('epoch')
plt.legend(loc='upper right') 
plt.show()
