import numpy as np
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
import tensorflow as tf
tf.random.set_seed(66)

#1. 데이터
datasets = load_iris()

x = datasets['data']
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split( x, y, train_size = 0.8, shuffle=True, random_state=68 )

#2. 모델구성
from sklearn.svm import LinearSVC,SVC
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn. ensemble import RandomForestClassifier

model = Perceptron(),SVC(),KNeighborsClassifier(),LogisticRegression(),DecisionTreeClassifier(),RandomForestClassifier()

for i in model:    
    model = i
    

    #3. 컴파일, 훈련
    model.fit(x_train, y_train)


    #4. 평가, 예측

    result = model.score(x_test,y_test)   

    y_predict = model.predict(x_test)

    print(f"{i} : ", round(result,4))   

    # Perceptron() :  0.7
'''    
SVC() :  0.9
KNeighborsClassifier() :  0.9667
LogisticRegression() :  0.9667
DecisionTreeClassifier() :  0.9667
RandomForestClassifier() :  0.9667
'''
