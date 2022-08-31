import numpy as np
from sklearn.model_selection import train_test_split,KFold, cross_val_score 
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import RobustScaler
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

datasets = fetch_california_housing() 
x = datasets.data 
y = datasets.target 
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7, shuffle=True, random_state=5)


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

for (name, algorithm) in allalgorithm : 
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
