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

model1 = DecisionTreeClassifier() 
model2 = RandomForestClassifier()
model3 = GradientBoostingClassifier()
model4 = XGBClassifier()

model_list = [ model1, model2, model3, model4 ]
model_name = ['DecisionTreeClassifier','RandomForestClassifier','XGBClassifier','GradientBoostingClassifier']

import matplotlib.pyplot as plt
def plot_feature_importances_dataset(model):
    n_features = datasets.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), datasets.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)
    
for i in range(4):
    # 3. 훈련
    plt.subplot(2,2,i+1)               
    model_list[i].fit(x_train, y_train)


    # 4. 평가, 예측
    result = model_list[i].score(x_train, y_train)
    feature_importances_ = model_list[i].feature_importances_

    from sklearn.metrics import accuracy_score
    y_predict = model_list[i].predict(x_test)
    acc = accuracy_score(y_test, y_predict)
    # print("result",result)
    # print("accuracy-score : ", acc)
    # print("feature_importances",feature_importances_)
    plot_feature_importances_dataset(model_list[i])
    plt.ylabel(model_name[i])



# plot_feature_importances_dataset(model)
plt.show()
