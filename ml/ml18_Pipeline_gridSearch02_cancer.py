
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target


x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.8,shuffle=True, random_state=1234)



#2. 모델구성
from sklearn.svm import LinearSVC, SVC
from sklearn. ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline, Pipeline


# model = SVC()
# model = make_pipeline(MinMaxScaler(),PCA(),RandomForestClassifier())  # piepline을 통해 순서대로 이동
pipe = Pipeline([('minmax', MinMaxScaler()), ('RF', RandomForestClassifier())], verbose=1)                                             

parameters = [
        {'RF__n_estimators':[100,200], 'RF__max_depth':[6,8,10,12], 'RF__min_samples_split':[2,3,5,10]},
        {'RF__n_estimators':[100,200], 'RF__min_samples_leaf':[3,5,7,10], 'RF__min_samples_split':[2,3,5,10]},
]

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle = True, random_state=66)
# gridsearch 사용 시 
# 언더바 2개 사용
# 언더바 2개 앞에는 내가 위에서 지정한 문자열, 뒤에는 하이퍼파라미터 이름 기재
                                            
#3. 훈련
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn. model_selection import HalvingGridSearchCV, HalvingRandomSearchCV

model = GridSearchCV(pipe, parameters, cv= kfold, verbose=1)

model.fit(x_train, y_train) 

#4. 평가, 예측
result = model.score(x_test, y_test)
print('[cancer]')
print('model.score:', round(result,4))         

                           