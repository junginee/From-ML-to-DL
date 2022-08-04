import numpy as np 
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.svm import LinearSVR
from sklearn.metrics import r2_score

#1.데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size=0.2,
                                                    shuffle=True,
                                                    random_state=66)

#2.모델구성
model = LinearSVR()


 
#3.컴파일,훈련
model.fit(x_train, y_train)


#4.평가,예측
print("보스톤")
results = model.score(x_test, y_test)
print("결과 : ", round(results,3)) 

y_predict = model.predict(x_test) 
r2 = r2_score(y_test, y_predict) #R2결정계수(성능평가지표)
print('r2스코어 : ' , round(r2,3)) 

# 보스톤
# 결과 :  0.746
# r2스코어 :  0.746

