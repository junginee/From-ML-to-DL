import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

#1. 데이터
datasets =load_diabetes()
x = datasets.data
y = datasets.target
x_train,x_test,y_train, y_test = train_test_split(x,y, train_size=0.75, shuffle=True, random_state=72)


scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)


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
      print(name, "의 정답률 : ", round(acc,5))
  except : 
    #   continue
    print(name, ": 미출력!!!!!!!!")
    
'''
AdaBoostClassifier 의 정답률 :  0.0
BaggingClassifier 의 정답률 :  0.0
BernoulliNB 의 정답률 :  0.00901
CalibratedClassifierCV 의 정답률 :  0.01802
CategoricalNB : 미출력!!!!!!!!
ClassifierChain : 미출력!!!!!!!!
ComplementNB : 미출력!!!!!!!!
DecisionTreeClassifier 의 정답률 :  0.00901
DummyClassifier 의 정답률 :  0.0
ExtraTreeClassifier 의 정답률 :  0.00901
ExtraTreesClassifier 의 정답률 :  0.00901
GaussianNB 의 정답률 :  0.0
GaussianProcessClassifier 의 정답률 :  0.00901
GradientBoostingClassifier 의 정답률 :  0.00901
HistGradientBoostingClassifier 의 정답률 :  0.0
KNeighborsClassifier 의 정답률 :  0.0
LabelPropagation 의 정답률 :  0.00901
LabelSpreading 의 정답률 :  0.00901
LinearDiscriminantAnalysis 의 정답률 :  0.00901
LinearSVC 의 정답률 :  0.00901
LogisticRegression 의 정답률 :  0.00901
LogisticRegressionCV : 미출력!!!!!!!!
MLPClassifier 의 정답률 :  0.0        
MultiOutputClassifier : 미출력!!!!!!!!
MultinomialNB : 미출력!!!!!!!!        
NearestCentroid 의 정답률 :  0.0     
NuSVC : 미출력!!!!!!!!
OneVsOneClassifier : 미출력!!!!!!!!  
OneVsRestClassifier : 미출력!!!!!!!! 
OutputCodeClassifier : 미출력!!!!!!!!
PassiveAggressiveClassifier 의 정답률 :  0.0
Perceptron 의 정답률 :  0.00901
QuadraticDiscriminantAnalysis : 미출력!!!!!!!!
RadiusNeighborsClassifier : 미출력!!!!!!!!
RandomForestClassifier 의 정답률 :  0.0
RidgeClassifier 의 정답률 :  0.01802
RidgeClassifierCV 의 정답률 :  0.01802
SGDClassifier 의 정답률 :  0.0
SVC 의 정답률 :  0.0
StackingClassifier : 미출력!!!!!!!!
VotingClassifier : 미출력!!!!!!!!
'''    
