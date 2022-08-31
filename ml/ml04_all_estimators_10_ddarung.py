import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.svm import LinearSVR # 레거시한 리니어 모델 사용
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score, r2_score
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
path = './_data/ddarung/' 
train_set = pd.read_csv(path + 'train.csv', 
                        index_col=0)


test_set = pd.read_csv(path + 'test.csv',
                       index_col=0)
submission = pd.read_csv(path + 'submission.csv',
                       index_col=0)


#### Missing value processing ####
train_set = train_set.fillna(train_set.mean()) # nan 값을 행별로 모두 삭제(dropna)
test_set = test_set.fillna(test_set.mean())


x = train_set.drop(['count'], axis=1) # axis는 'count'가 컬럼이라는 것을 명시하기 위해
y = train_set['count']


x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.98, shuffle=True, random_state=68)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델구성
allAlgorithms = all_estimators(type_filter='regressor')
print('모델의 갯수 :', len(allAlgorithms)) # 모델의 갯수 : 41



for (name, algorithm) in allAlgorithms:
    try:
        model = algorithm()
        model.fit(x_train, y_train)
    
        y_predict = model.predict(x_test)
        acc = r2_score(y_test, y_predict)
        print(name, '의 정답률 :', acc)
    except:
        # continue
        print(name, '은 안나온 애들!')

#3. 컴파일, 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test) # evaluate 대신 score 사용
print('결과 :', result)

from sklearn.metrics import r2_score
y_predict_1 = model.predict(x_test)
acc = r2_score(y_test, y_predict_1)
print('r2 스코어 : ', acc)

# 결과 : 0.11906345565855969
# r2 스코어 :  0.11906345565855969
