import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

#1. 데이터
datasets = load_breast_cancer()

x = datasets.data   
y = datasets.target
print(x.shape, y.shape) 


x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.7,shuffle=True, random_state=72)
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

for (name, algorithm) in allalgorithm : 
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
AdaBoostClassifier 의 정답률 :  0.9766
BaggingClassifier 의 정답률 :  0.9357
BernoulliNB 의 정답률 :  0.8889
CalibratedClassifierCV 의 정답률 :  0.9532
CategoricalNB : 미출력!!!!!!!!
ClassifierChain : 미출력!!!!!!!!
ComplementNB : 미출력!!!!!!!!
DecisionTreeClassifier 의 정답률 :  0.8889
DummyClassifier 의 정답률 :  0.614
ExtraTreeClassifier 의 정답률 :  0.8713
ExtraTreesClassifier 의 정답률 :  0.9766
GaussianNB 의 정답률 :  0.9123
GaussianProcessClassifier 의 정답률 :  0.9649
GradientBoostingClassifier 의 정답률 :  0.9474
HistGradientBoostingClassifier 의 정답률 :  0.9532
KNeighborsClassifier 의 정답률 :  0.9591
LabelPropagation 의 정답률 :  0.9123
LabelSpreading 의 정답률 :  0.9123
LinearDiscriminantAnalysis 의 정답률 :  0.9357
LinearSVC 의 정답률 :  0.9591
LogisticRegression 의 정답률 :  0.9708
LogisticRegressionCV 의 정답률 :  0.9766
MLPClassifier 의 정답률 :  0.9649
MultiOutputClassifier : 미출력!!!!!!!!
MultinomialNB : 미출력!!!!!!!!
NearestCentroid 의 정답률 :  0.9123
NuSVC 의 정답률 :  0.9298
OneVsOneClassifier : 미출력!!!!!!!!
OneVsRestClassifier : 미출력!!!!!!!!
OutputCodeClassifier : 미출력!!!!!!!!
PassiveAggressiveClassifier 의 정답률 :  0.9357
Perceptron 의 정답률 :  0.9415
QuadraticDiscriminantAnalysis 의 정답률 :  0.9532
RadiusNeighborsClassifier : 미출력!!!!!!!!
RandomForestClassifier 의 정답률 :  0.9649
RidgeClassifier 의 정답률 :  0.9415
RidgeClassifierCV 의 정답률 :  0.9415
SGDClassifier 의 정답률 :  0.9415
SVC 의 정답률 :  0.9766
StackingClassifier : 미출력!!!!!!!!
VotingClassifier : 미출력!!!!!!!!
'''    
