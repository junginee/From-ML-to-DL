import numpy as np
import tensorflow as tf
from sklearn. datasets import fetch_covtype 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from sklearn.svm import LinearSVC
from sklearn.utils import all_estimators


#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y= datasets.target

# print(datasets.feature_names)
# print(datasets.DESCR)
# print(x.shape, y.shape) #(581012, 54) (581012,)
# print(np.unique(y, return_counts = True)) # y :[1 2 3 4 5 6 7]  / return_counts :[211840, 283301,  35754,   2747,   9493,  17367,  20510]

x_train, x_test, y_train, y_test = train_test_split( x, y, train_size = 0.8, shuffle=True, random_state=68 )

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
