from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from xgboost import XGBClassifier, XGBRegressor, XGBRFRegressor, XGBRFClassifier
from sklearn.metrics import accuracy_score, r2_score
import numpy as np
import time



#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
print(x.shape) # (569, 30)
print(y.shape) # (569,)

x_train, x_test, y_train, y_test = train_test_split(x,y,
        train_size=0.8, shuffle=True, random_state=123, stratify=y)

#2. 모델
model = XGBClassifier(n_estimators = 200, learning_rate = 0.15, max_depth = 5, gamma = 0, min_child_weight = 0.5, random_state=123)
    
#3. 컴파일,훈련
start = time.time()
model.fit(x_train,y_train, early_stopping_rounds=10, 
          eval_set=[(x_train,y_train), (x_test,y_test)], eval_metric='error', verbose=1)
        #   eval_set=[(x_test,y_test)])
end = time.time()- start


#4. 평가, 예측
result = model.score(x_test, y_test)

print('model.score : ', result)

y_predict = model.predict(x_test)

print('accuracy_score :',accuracy_score(y_test,y_predict))

# import joblib
path = 'D:\study_data\_save\_xg\\'
# joblib.dump(model, path+'m40_joblib1_save.dat')
# pickle, joblib 모두 저장 가능하다.(원래는 파이썬 바이너리 세이브용)

model.save_model(path + 'ml41_xgb1_save.dat')
