import numpy as np
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split 
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

#1. 데이터
datasets = load_iris()

x = datasets['data']
y = datasets.target


print(x.shape, y.shape) 
print("y의 라벨값(y의 고유값)", np.unique(y)) #y의 라벨값(y의 고유값) [0 1 2]


x_train, x_test, y_train, y_test = train_test_split( x, y, train_size = 0.8, shuffle=True, random_state=68 )

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


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
    
'''
AdaBoostClassifier 의 정답률 :  0.9667
BaggingClassifier 의 정답률 :  0.9667
BernoulliNB 의 정답률 :  0.3333
CalibratedClassifierCV 의 정답률 :  0.9667
CategoricalNB 의 정답률 :  0.3333
ClassifierChain : 미출력!!!!!!!!
ComplementNB 의 정답률 :  0.7
DecisionTreeClassifier 의 정답률 :  1.0
DummyClassifier 의 정답률 :  0.3
ExtraTreeClassifier 의 정답률 :  0.9333
ExtraTreesClassifier 의 정답률 :  0.9667
GaussianNB 의 정답률 :  0.9667
GaussianProcessClassifier 의 정답률 :  0.9667
GradientBoostingClassifier 의 정답률 :  0.9667
HistGradientBoostingClassifier 의 정답률 :  0.9667
KNeighborsClassifier 의 정답률 :  0.9667
LabelPropagation 의 정답률 :  0.9667
LabelSpreading 의 정답률 :  0.9333
LinearDiscriminantAnalysis 의 정답률 :  0.9667
LinearSVC 의 정답률 :  0.9667
LogisticRegression 의 정답률 :  0.9667
LogisticRegressionCV 의 정답률 :  0.9667
MLPClassifier 의 정답률 :  0.9667
MultiOutputClassifier : 미출력!!!!!!!!
MultinomialNB 의 정답률 :  0.6
NearestCentroid 의 정답률 :  0.9333
NuSVC 의 정답률 :  0.9667
OneVsOneClassifier : 미출력!!!!!!!!
OneVsRestClassifier : 미출력!!!!!!!!
OutputCodeClassifier : 미출력!!!!!!!!
PassiveAggressiveClassifier 의 정답률 :  0.9333
Perceptron 의 정답률 :  0.8333
QuadraticDiscriminantAnalysis 의 정답률 :  0.9667
RadiusNeighborsClassifier 의 정답률 :  0.3667
RandomForestClassifier 의 정답률 :  0.9667
RidgeClassifier 의 정답률 :  0.9
RidgeClassifierCV 의 정답률 :  0.9
SGDClassifier 의 정답률 :  0.9333
SVC 의 정답률 :  0.9667
StackingClassifier : 미출력!!!!!!!!
VotingClassifier : 미출력!!!!!!!!
'''    
  
