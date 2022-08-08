import numpy as np
from sklearn.datasets import load_iris

#1. 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test = train_test_split(x,y,
        train_size=0.8,shuffle=True, random_state=123)

#2. 모델구성
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# model = DecisionTreeClassifier() 
# model = RandomForestClassifier()
# model = GradientBoostingClassifier()
model = XGBClassifier()
#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test)
print("model.score : ", round(result,4))

from sklearn.metrics import accuracy_score
y_predict = model.predict(x_test)
acc = accuracy_score(y_test, y_predict)
print("accuacy score : ", round(acc,4))

print("=======================")
print(model, ':', model.feature_importances_) #feature 중요도를 알 수 有

# DecisionTreeClassifier() # [0.01088866 0.01253395 0.54516978 0.43140761]
# RandomForestClassifier() : [0.07651702 0.02087412 0.44554124 0.45706762]
# GradientBoostingClassifier() : [0.00080989 0.02430213 0.56013843 0.41474955]
# XGBClassifier() : [0.0089478  0.01652037 0.75273126 0.22180054]
