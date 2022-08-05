import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split,KFold, cross_val_score 
from sklearn. datasets import load_digits
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

#1. 데이터
datasets = load_digits()
x = datasets.data
y= datasets.target

print(x.shape, y.shape)
print(np.unique(y)) 

import tensorflow as tf
tf.random.set_seed(66)

x_train, x_test, y_train, y_test = train_test_split( x, y, train_size = 0.8, shuffle=True, random_state=68 )

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)
print(np.min(x_train)) 
print(np.max(x_train))

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

'''
kfold_all_estimators
AdaBoostClassifier 0.2738
BaggingClassifier 0.9293
BernoulliNB 0.8514
CalibratedClassifierCV 0.9622
CategoricalNB : 미출력!!!!!!!!
ClassifierChain : 미출력!!!!!!!!
ComplementNB : 미출력!!!!!!!!
DecisionTreeClassifier 0.8553
DummyClassifier 0.0785
ExtraTreeClassifier 0.7919
ExtraTreesClassifier 0.985
GaussianNB 0.8392
GaussianProcessClassifier 0.1046
GradientBoostingClassifier 0.9644
HistGradientBoostingClassifier 0.9727
KNeighborsClassifier 0.9883
LabelPropagation 0.1002
LabelSpreading 0.1002
LinearDiscriminantAnalysis 0.9533
LinearSVC 0.9549
LogisticRegression 0.9716
LogisticRegressionCV 0.9705
MLPClassifier 0.9727
MultiOutputClassifier : 미출력!!!!!!!!
MultinomialNB : 미출력!!!!!!!!
NearestCentroid 0.8998
NuSVC 0.9616
OneVsOneClassifier : 미출력!!!!!!!!
OneVsRestClassifier : 미출력!!!!!!!!
OutputCodeClassifier : 미출력!!!!!!!!
PassiveAggressiveClassifier 0.9477
Perceptron 0.9466
QuadraticDiscriminantAnalysis 0.8437
RadiusNeighborsClassifier : 미출력!!!!!!!!
RandomForestClassifier 0.9777
RidgeClassifier 0.9338
RidgeClassifierCV 0.9343
SGDClassifier 0.951
SVC 0.9883
StackingClassifier : 미출력!!!!!!!!
VotingClassifier : 미출력!!!!!!!!
'''