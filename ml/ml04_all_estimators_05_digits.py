import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn. datasets import load_digits
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

#1. 데이터
datasets = load_digits()
x = datasets.data
y= datasets.target

# print(x.shape, y.shape)
# print(np.unique(y)) 

import tensorflow as tf
tf.random.set_seed(66)

x_train, x_test, y_train, y_test = train_test_split( x, y, train_size = 0.8, shuffle=True, random_state=68 )

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)
print(np.min(x_train)) 
print(np.max(x_train))


#2. 모델구성

allalgorithm = all_estimators(type_filter='classifier')

print('allalgorithms : ', allalgorithm)
print("모델의 갯수 : ", len(allalgorithm)) #모델의 갯수 :  41

for (name, algorithm) in allalgorithm : 
  try : 
      model = algorithm()
      model.fit(x_train, y_train)
 
      y_predict = model.predict(x_test)
      acc = accuracy_score(y_test,y_predict)
      print(name, "의 정답률 : ", round(acc,4))
  except : 
    print(name, ": 미출력!!!!!!!!")

'''
AdaBoostClassifier 의 정답률 :  0.2639
BaggingClassifier 의 정답률 :  0.9083
BernoulliNB 의 정답률 :  0.875
CalibratedClassifierCV 의 정답률 :  0.9778
CategoricalNB : 미출력!!!!!!!!
ClassifierChain : 미출력!!!!!!!!
ComplementNB : 미출력!!!!!!!!
DecisionTreeClassifier 의 정답률 :  0.8389
DummyClassifier 의 정답률 :  0.075
ExtraTreeClassifier 의 정답률 :  0.8139
ExtraTreesClassifier 의 정답률 :  0.9806
GaussianNB 의 정답률 :  0.8139
GaussianProcessClassifier 의 정답률 :  0.9361
GradientBoostingClassifier 의 정답률 :  0.975
HistGradientBoostingClassifier 의 정답률 :  0.9806
KNeighborsClassifier 의 정답률 :  0.9056
LabelPropagation 의 정답률 :  0.9278
LabelSpreading 의 정답률 :  0.9278
LinearDiscriminantAnalysis 의 정답률 :  0.95
LinearSVC 의 정답률 :  0.9639
LogisticRegression 의 정답률 :  0.9722
LogisticRegressionCV 의 정답률 :  0.9694
MLPClassifier 의 정답률 :  0.9722
MultiOutputClassifier : 미출력!!!!!!!!
MultinomialNB : 미출력!!!!!!!!
NearestCentroid 의 정답률 :  0.7028
NuSVC 의 정답률 :  0.9167
OneVsOneClassifier : 미출력!!!!!!!!
OneVsRestClassifier : 미출력!!!!!!!!
OutputCodeClassifier : 미출력!!!!!!!!
PassiveAggressiveClassifier 의 정답률 :  0.9583
Perceptron 의 정답률 :  0.9528
QuadraticDiscriminantAnalysis 의 정답률 :  0.8639
RadiusNeighborsClassifier : 미출력!!!!!!!!
RandomForestClassifier 의 정답률 :  0.9694
RidgeClassifier 의 정답률 :  0.9361
RidgeClassifierCV 의 정답률 :  0.9417
SGDClassifier 의 정답률 :  0.9611
SVC 의 정답률 :  0.975
StackingClassifier : 미출력!!!!!!!!
VotingClassifier : 미출력!!!!!!!!
'''
