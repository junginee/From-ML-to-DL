import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.datasets import load_wine 
from sklearn.model_selection import train_test_split ,KFold, cross_val_score 
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
import tensorflow as tf
tf.random.set_seed(66)
import warnings
warnings.filterwarnings("ignore")

#1. 데이터
datasets = load_wine()
x = datasets.data
y= datasets.target

print(x.shape, y.shape) #(178, 13) #(178,)
print(np.unique(y, return_counts = True)) #[0 1 2]

x_train, x_test, y_train, y_test = train_test_split( x, y, train_size = 0.8, shuffle=True, random_state=68 )

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)
print(np.min(x_train)) #0.0 
print(np.max(x_train)) #1.0

n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle = True, random_state=66)

#2. 모델구성

allalgorithm = all_estimators(type_filter='classifier')

print('allalgorithms : ', allalgorithm)
print("모델의 갯수 : ", len(allalgorithm)) #모델의 갯수 :  41
print('kfold_all_estimators')
for (name, algorithm) in allalgorithm : #name-algorithm : key-value 쌍으로 이루는 dictionary
  try : 
      model = algorithm()
      model.fit(x_train, y_train)
 
      y_predict = model.predict(x_test)
      scores = cross_val_score(model,x,y,cv=kfold)
      print(name,
            # scores,'\ncross_val_score :', 
            round(np.mean(scores),4))
  except : 
    #   continue
    print(name, ": 미출력!!!!!!!!")
    
# kfold_all_estimators
# AdaBoostClassifier 0.9106
# BaggingClassifier 0.9721
# BernoulliNB 0.399
# CalibratedClassifierCV 0.9156
# CategoricalNB : 미출력!!!!!!!!
# ClassifierChain : 미출력!!!!!!!!
# ComplementNB : 미출력!!!!!!!!
# DecisionTreeClassifier 0.9325
# DummyClassifier 0.399
# ExtraTreeClassifier 0.9103
# ExtraTreesClassifier 0.9832
# GaussianNB 0.9721
# GaussianProcessClassifier 0.4783
# GradientBoostingClassifier 0.9441
# HistGradientBoostingClassifier 0.9776
# KNeighborsClassifier 0.691
# LabelPropagation 0.4886
# LabelSpreading 0.4886
# LinearDiscriminantAnalysis 0.9887
# LinearSVC 0.8378
# LogisticRegression 0.9608
# LogisticRegressionCV 0.9551
# MLPClassifier 0.6779
# MultiOutputClassifier : 미출력!!!!!!!!
# MultinomialNB : 미출력!!!!!!!!
# NearestCentroid 0.7251
# NuSVC 0.8703
# OneVsOneClassifier : 미출력!!!!!!!!
# OneVsRestClassifier : 미출력!!!!!!!!
# OutputCodeClassifier : 미출력!!!!!!!!
# PassiveAggressiveClassifier 0.6456
# Perceptron 0.6006
# QuadraticDiscriminantAnalysis 0.9944
# RadiusNeighborsClassifier : 미출력!!!!!!!!
# RandomForestClassifier 0.9832
# RidgeClassifier 0.9943
# RidgeClassifierCV 0.9943
# SGDClassifier 0.5892
# SVC 0.6457
# StackingClassifier : 미출력!!!!!!!!
# VotingClassifier : 미출력!!!!!!!!
