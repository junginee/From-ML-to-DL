import numpy as np 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split, KFold, cross_val_score 
from sklearn.datasets import load_boston
from sklearn.svm import LinearSVR
from sklearn.metrics import r2_score
import tensorflow as tf
tf.random.set_seed(66)

#1.데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size=0.2,
                                                    shuffle=True)


n_splits = 5
kfold = KFold(n_splits=n_splits, shuffle = True, random_state=100)                                                    random_state=66)


#2.모델구성
model = LinearSVR()


 
#3,4. 컴파일, 훈련, 평가, 예측

# model.fit(x_train, y_train)
scores = cross_val_score(model,x_train,y_train,cv=kfold)
print('ACC : ',scores,'\ncross_val_score :', round(np.mean(scores),4))

