import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.datasets import load_wine 
from sklearn.model_selection import train_test_split 
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

#1. 데이터
datasets = load_wine()
x = datasets.data
y= datasets.target

print(x.shape, y.shape) #(178, 13) #(178,)
print(np.unique(y, return_counts = True)) #[0 1 2]

import tensorflow as tf
tf.random.set_seed(66)

x_train, x_test, y_train, y_test = train_test_split( x, y, train_size = 0.8, shuffle=True, random_state=68 )

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)
print(np.min(x_train)) #0.0 
print(np.max(x_train)) #1.0

#2. 모델구성

allalgorithm = all_estimators(type_filter='classifier')

print('allalgorithms : ', allalgorithm)
print("모델의 갯수 : ", len(allalgorithm)) #모델의 갯수 :  41

for (name, algorithm) in allalgorithm : #name-algorithm : key-value 쌍으로 이루는 dictionary
  try : 
      model = algorithm()
      model.fit(x_train, y_train)
 
      y_predict = model.predict(x_test)
      acc = accuracy_score(y_test,y_predict)
      print(name, "의 정답률 : ", round(acc,4))
  except : 
    #   continue
    print(name, ": 미출력!!!!!!!!")
    
# AdaBoostClassifier 의 정답률 :  0.8611
# BaggingClassifier 의 정답률 :  0.9444
# BernoulliNB 의 정답률 :  0.9167
# CalibratedClassifierCV 의 정답률 :  0.9444
# CategoricalNB : 미출력!!!!!!!!
# ClassifierChain : 미출력!!!!!!!!
# ComplementNB : 미출력!!!!!!!!
# DecisionTreeClassifier 의 정답률 :  0.8611
# DummyClassifier 의 정답률 :  0.4444
# ExtraTreeClassifier 의 정답률 :  0.7778
# ExtraTreesClassifier 의 정답률 :  0.9722
# GaussianNB 의 정답률 :  0.9722
# GaussianProcessClassifier 의 정답률 :  0.9444
# GradientBoostingClassifier 의 정답률 :  0.8889
# HistGradientBoostingClassifier 의 정답률 :  0.9444
# KNeighborsClassifier 의 정답률 :  0.9444
# LabelPropagation 의 정답률 :  0.9444
# LabelSpreading 의 정답률 :  0.9444
# LinearDiscriminantAnalysis 의 정답률 :  0.9444
# LinearSVC 의 정답률 :  0.9444
# LogisticRegression 의 정답률 :  0.9167
# LogisticRegressionCV 의 정답률 :  0.9167
# MLPClassifier 의 정답률 :  0.9167
# MultiOutputClassifier : 미출력!!!!!!!!
# MultinomialNB : 미출력!!!!!!!!
# NearestCentroid 의 정답률 :  0.9444
# NuSVC 의 정답률 :  0.9722
# OneVsOneClassifier : 미출력!!!!!!!!
# OneVsRestClassifier : 미출력!!!!!!!!
# OutputCodeClassifier : 미출력!!!!!!!!
# PassiveAggressiveClassifier 의 정답률 :  0.9444
# Perceptron 의 정답률 :  0.9167
# QuadraticDiscriminantAnalysis 의 정답률 :  0.9722
# RadiusNeighborsClassifier : 미출력!!!!!!!!
# RandomForestClassifier 의 정답률 :  0.9444
# RidgeClassifier 의 정답률 :  1.0
# RidgeClassifierCV 의 정답률 :  1.0
# SGDClassifier 의 정답률 :  0.9722
# SVC 의 정답률 :  1.0
# StackingClassifier : 미출력!!!!!!!!
# VotingClassifier : 미출력!!!!!!!!