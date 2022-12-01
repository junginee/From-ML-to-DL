import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from sklearn.metrics import r2_score

datasets =load_diabetes()
x = datasets.data
y = datasets.target
x_train,x_test,y_train, y_test = train_test_split(x,y, train_size=0.75, shuffle=True, random_state=72)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)     
x_test = scaler.transform(x_test)
print(np.min(x_train)) #0.0 
print(np.max(x_train)) #1.0


#2.모델구성
model = LinearSVR() 


#3. 컴파일, 훈련     
model.fit(x_train, y_train)                        


#4.평가, 예측
results = model.score(x_test, y_test)
print("결과 : ", round(results,3)) 
y_predict = model.predict(x_test) 

r2 = r2_score(y_test, y_predict) 
print('r2스코어 : ' , round(r2,3)) 

# 결과 :  0.348
# r2스코어 :  0.348
