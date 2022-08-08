import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

#1. 데이터
datasets =load_diabetes()
x = datasets.data
y = datasets.target


x_train, x_test, y_train,y_test = train_test_split(x,y,
        train_size=0.8,shuffle=True, random_state=123)

#2. 모델구성
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.ensemble import  GradientBoostingClassifier, GradientBoostingRegressor
from xgboost import XGBClassifier, XGBRegressor

# model = DecisionTreeRegressor() 
# model = RandomForestRegressor()
# model = GradientBoostingRegressor()
model = XGBRegressor()

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
result = model.score(x_test, y_test)
print("model.score : ", round(result,4))

from sklearn.metrics import accuracy_score, r2_score
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)
print("r2 : ", round(r2,4))

print("=======================")
print(model, ':', model.feature_importances_) #feature 중요도를 알 수 有

# model.score :  0.1482
# DecisionTreeRegressor() : [0.0897716  0.0230252  0.2339448  0.05252239 0.04966723 0.06435879
#  0.04145477 0.00953339 0.36244456 0.07327727]


# model.score :  0.5297
# RandomForestRegressor() : [0.06104859 0.01119429 0.28949376 0.10020903 0.03927626 0.05494114
#  0.05710663 0.02652076 0.28457684 0.07563269]


# model.score :  0.5573
# GradientBoostingRegressor() : [0.04966131 0.01077999 0.30315057 0.11149356 0.02814258 0.05739129
#  0.04053694 0.01580331 0.33863899 0.04440145]


# model.score :  0.459
# XGBRegressor : [0.03234756 0.0447546  0.21775807 0.08212128 0.04737141 0.04843819      
#  0.06012432 0.09595273 0.30483875 0.06629313]