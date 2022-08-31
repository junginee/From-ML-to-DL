import numpy as np
from sklearn. datasets import load_digits
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터
datasets = load_digits()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
        train_size=0.8,shuffle=True, random_state=1234)

# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

#2. 모델구성
from sklearn.svm import LinearSVC, SVC
from sklearn. ensemble import RandomForestClassifier

from sklearn.pipeline import make_pipeline

#model = SVC()
model = make_pipeline(MinMaxScaler(),RandomForestClassifier())                                    
                                             
#3. 훈련
model.fit(x_train, y_train) 

#4. 평가, 예측
result = model.score(x_test, y_test)
print('digits')
print('model.score:', round(result,4))   

# digits
# model.score: 0.9694                                        
