import numpy as np
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split 
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score 
import tensorflow as tf           
tf.random.set_seed(66)       
         
#1. 데이터         
datasets = load_iris()       
   
x = datasets['data']        
y = datasets.target


print(x.shape, y.shape) 
print("y의 라벨값(y의 고유값)", np.unique(y)) #y의 라벨값(y의 고유값) [0 1 2]

# from tensorflow.keras.utils import to_categorical 
# y = to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split( x, y, train_size = 0.8, shuffle=True, random_state=68 )

#2. 모델구성

model = LinearSVC() #linearsvc 모델은 원핫인코딩 사용x

#3. 컴파일, 훈련

model.fit(x_train, y_train)

#4. 평가, 예측

results = model.score(x_test, y_test)

print("결과 acc : ", round(results,3)) 

from sklearn.metrics import r2_score, accuracy_score  
y_predict = model.predict(x_test)

acc= accuracy_score(y_test, y_predict)
print('acc스코어 : ', round(acc,3)) 

# 결과 acc :  0.967
# acc스코어 :  0.967
