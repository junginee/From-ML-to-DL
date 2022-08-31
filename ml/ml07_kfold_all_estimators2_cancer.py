import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split,KFold, cross_val_score 
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
AdaBoostClassifier 0.9649
BaggingClassifier 0.9385
BernoulliNB 0.6274
CalibratedClassifierCV 0.9263
CategoricalNB : 미출력!!!!!!!!
ClassifierChain : 미출력!!!!!!!!
ComplementNB : 미출력!!!!!!!!
DecisionTreeClassifier 0.9192
DummyClassifier 0.6274
ExtraTreeClassifier 0.9227
ExtraTreesClassifier 0.9684
GaussianNB 0.942
GaussianProcessClassifier 0.9122
GradientBoostingClassifier 0.9578
HistGradientBoostingClassifier 0.9737
KNeighborsClassifier 0.928
LabelPropagation 0.3902
LabelSpreading 0.3902
LinearDiscriminantAnalysis 0.9614
LinearSVC 0.8946
LogisticRegression 0.9367
LogisticRegressionCV 0.9543
MLPClassifier 0.9297
MultiOutputClassifier : 미출력!!!!!!!!
MultinomialNB : 미출력!!!!!!!!        
NearestCentroid 0.8893
NuSVC 0.8735
OneVsOneClassifier : 미출력!!!!!!!!  
OneVsRestClassifier : 미출력!!!!!!!! 
OutputCodeClassifier : 미출력!!!!!!!!
PassiveAggressiveClassifier 0.9069   
Perceptron 0.7771
QuadraticDiscriminantAnalysis 0.9525      
RadiusNeighborsClassifier : 미출력!!!!!!!!
RandomForestClassifier 0.9614
RidgeClassifier 0.9543       
RidgeClassifierCV 0.9561
SGDClassifier 0.928
SVC 0.921
StackingClassifier : 미출력!!!!!!!!
VotingClassifier : 미출력!!!!!!!!  
'''    
