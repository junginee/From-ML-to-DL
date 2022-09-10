# [조건 1]
# 0 212 => 라벨 0 112개 삭제 후 증폭
# 1 357

# [조건 2]
# smote 넣기 전, 후

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE


dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

zero_index = []
break_num = 0

while break_num < 112 :
 for i in range(len(y)) :
     if y[i] == 0:
      zero_index.append(i)
      break_num += 1
      break
  
print(len(zero_index)) #112


print(np.unique(y, return_counts=True)) # [3. 4. 5. 6. 7. 8. 9.]
# le = LabelEncoder()
# y = le.fit_transform(y)
# print(np.unique(y, return_counts=True)) # [0 1 2 3 4 5 6]

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=123,
                                                    shuffle=True, stratify=y)

smote = SMOTE(random_state=123, k_neighbors=1)
x_train, y_train = smote.fit_resample(x_train, y_train, )

print(np.unique(y_train, return_counts=True)) # [3. 4. 5. 6. 7. 8. 9.]


#2. 모델
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression # 이진분류모델

model = RandomForestClassifier()

#3. 컴파일, 훈련

model.fit(x_train, y_train)

#4. 평가, 예측
y_predict = model.predict(x_test)

score = model.score(x_test, y_test)

from sklearn.metrics import accuracy_score, f1_score
print('model.score :' , round(score,4))
print("accuracy score : ",
      round(accuracy_score(y_test, y_predict), 4 ))
print("f1_score(macro) : ",round(f1_score(y_test, y_predict, average = 'macro'),4))
print("f1_score(micro) : ",round(f1_score(y_test, y_predict, average = 'micro'),4))

###################smote 적용 전###################
# model.score : 0.9649
# accuracy score :  0.9649
# f1_score(macro) :  0.9627
# f1_score(micro) :  0.9649

###################smote 적용 후###################
# model.score : 0.9737
# accuracy score :  0.9737     
# f1_score(macro) :  0.9721
# f1_score(micro) :  0.9737
