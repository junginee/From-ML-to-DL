import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split,KFold, cross_val_score 
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

#1.데이터
path = './_data/ddarung/'
train_set = pd.read_csv(path + 'train.csv', index_col =0) 

print(train_set)
print(train_set.shape) #(1459,10)

test_set = pd.read_csv(path + 'test.csv', index_col =0) 
print(test_set)
print(test_set.shape)  #(715, 9)

print(train_set.columns)
print(train_set.info())  
print(train_set.describe())

print(train_set.isnull().sum())
train_set = train_set.dropna() 
print(train_set.isnull().sum())
x = train_set.drop(['count'], axis = 1) 

print(x)
print(x.columns)
print(x.shape) #(1459,9)

y = train_set['count'] #count 컬럼만 y 변수에 저장한다.
print(y)
print(y.shape)

x_train,x_test,y_train, y_test = train_test_split(x,y, train_size=0.7, shuffle=True, random_state=72)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)
test_set = scaler.transform(test_set)
print(np.min(x_train))  # 0.0
print(np.max(x_train))  # 1.0

print(np.min(x_test))  # 1.0
print(np.max(x_test))  # 1.0

#2. 모델구성

allalgorithm = all_estimators(type_filter='classifier')

print('allalgorithms : ', allalgorithm)
print("모델의 갯수 : ", len(allalgorithm)) #모델의 갯수 :  41
print('kfold_all_estimators')
for (name, algorithm) in allalgorithm : 
  try : 
      model = algorithm()
      model.fit(x_train, y_train)
 
      y_predict = model.predict(x_test)
      print(name,
            # scores,'\ncross_val_score :', 
            round(np.mean(scores),4))
  except : 
    #   continue
    print(name, ": 미출력!!!!!!!!")
