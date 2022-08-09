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
from xgboost import XGBRegressor
model = XGBRegressor()

#3. 훈련
model.fit(x_train, y_train)
print(model, ':', model.feature_importances_) #feature 중요도를 알 수 有

import matplotlib.pyplot as plt
from xgboost.plotting import plot_importance
plot_importance(model)
plt.show()