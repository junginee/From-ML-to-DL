from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import r2_score, accuracy_score
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
import time

###########################폴더 생성시 현재 파일명으로 자동생성###########################################
import inspect, os
a = inspect.getfile(inspect.currentframe()) #현재 파일이 위치한 경로 + 현재 파일 명
print(a)
print(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))) #현재 파일이 위치한 경로
print(a.split("\\")[-1]) #현재 파일 명
current_name = a.split("\\")[-1]
##########################밑에 filepath경로에 추가로  + current_name + '/' 삽입해야 돌아감#######################


#1. 데이터
path = './_data/kaggle_titanic/'
train_set = pd.read_csv(path + 'train.csv', # + 명령어는 문자를 앞문자와 더해줌
                        index_col=0) # index_col=n n번째 컬럼을 인덱스로 인식
# print(train_set)
# print(train_set.shape) # (891, 11)
# print(train_set.describe())
# print(train_set.columns)

test_set = pd.read_csv(path + 'test.csv', # 예측에서 쓸거임                
                       index_col=0)
# print(test_set)
# print(test_set.shape) # (418, 10)
# print(test_set.describe())

print(train_set.Pclass.value_counts())

Pclass1 = train_set["Survived"][train_set["Pclass"] == 1].value_counts(normalize = True)[1]*100
Pclass2 = train_set["Survived"][train_set["Pclass"] == 2].value_counts(normalize = True)[1]*100
Pclass3 = train_set["Survived"][train_set["Pclass"] == 3].value_counts(normalize = True)[1]*100
print(f"Percentage of Pclass 1 who survived: {Pclass1}")
print(f"Percentage of Pclass 2 who survived: {Pclass2}")
print(f"Percentage of Pclass 3 who survived: {Pclass3}")

female = train_set["Survived"][train_set["Sex"] == 'female'].value_counts(normalize = True)[1]*100
male = train_set["Survived"][train_set["Sex"] == 'male'].value_counts(normalize = True)[1]*100
print(f"Percentage of females who survived: {female}")
print(f"Percentage of males who survived: {male}")

sns.barplot(x="SibSp", y="Survived", data=train_set)


# df = pd.DataFrame(y)
# print(df)
# oh = OneHotEncoder(sparse=False) # sparse=true 는 매트릭스반환 False는 array 반환
# y = oh.fit_transform(df)
# print(y)



# print(test_set.columns)
# print(train_set.info()) # info 정보출력
# print(train_set.describe()) # describe 평균치, 중간값, 최소값 등등 출력

#### 결측치 처리 1. 제거 ####

train_set = train_set.fillna({"Embarked": "C"})
train_set.Age = train_set.Age.fillna(value=train_set.Age.mean())

train_set = train_set.drop(['Name'], axis = 1)
test_set = test_set.drop(['Name'], axis = 1)

train_set = train_set.drop(['Ticket'], axis = 1)
test_set = test_set.drop(['Ticket'], axis = 1)

train_set = train_set.drop(['Cabin'], axis = 1)
test_set = test_set.drop(['Cabin'], axis = 1)

train_set = pd.get_dummies(train_set,drop_first=True)
test_set = pd.get_dummies(test_set,drop_first=True)

test_set.Age = test_set.Age.fillna(value=test_set.Age.mean())
test_set.Fare = test_set.Fare.fillna(value=test_set.Fare.mode())

print(train_set, test_set, train_set.shape, test_set.shape)

############################


x = train_set.drop(['Survived'], axis=1)  # drop 데이터에서 ''사이 값 빼기
print(x)
print(x.columns)
print(x.shape) # (891, 8)

y = train_set['Survived'] 
print(y)
print(y.shape) # (891,)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=66
                                                    )

# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
test_set = scaler.transform(test_set)
print(np.min(x_train))  # 0.0
print(np.max(x_train))  # 1.0

print(np.min(x_test))  # 1.0
print(np.max(x_test))  # 1.0


#2. 모델구성

# model = load_model("./_save/keras22_hamsu12_kaggle_titanic.h5")

# model = Sequential()
# model.add(Dense(300, input_dim=8, activation='relu')) #sigmoid : 이진분류일때 아웃풋에 activation = 'sigmoid' 라고 넣어줘서 아웃풋 값 범위를 0에서 1로 제한해줌
# model.add(Dense(200, activation='relu'))               # 출력이 0 or 1으로 나와야되기 때문, 그리고 최종으로 나온 값에 반올림을 해주면 0 or 1 완성
# model.add(Dense(200, activation='relu'))               # relu : 히든에서만 쓸수있음, 요즘에 성능 젤좋음
# model.add(Dense(200, activation='relu'))
# model.add(Dense(200, activation='relu'))
# model.add(Dense(200, activation='relu'))
# model.add(Dense(200, activation='relu'))
# model.add(Dense(200, activation='relu'))
# model.add(Dense(200, activation='relu'))
# model.add(Dense(200, activation='relu'))               
# model.add(Dense(1, activation='sigmoid'))   

input1 = Input(shape=(8,))
dense1 = Dense(300, activation='relu')(input1)
dense2 = Dense(200, activation='relu')(dense1)
dense3 = Dense(200, activation='relu')(dense2)
dense4 = Dense(200, activation='relu')(dense3)
dense5 = Dense(200, activation='relu')(dense4)
dense6 = Dense(200, activation='relu')(dense5)
dense7 = Dense(200, activation='relu')(dense6)
dense8 = Dense(200, activation='relu')(dense7)
dense9 = Dense(200, activation='relu')(dense8)
dense10 = Dense(200, activation='relu')(dense9)
output1 = Dense(1, activation='sigmoid')(dense10)
model = Model(inputs=input1, outputs=output1) 
                                                                        
#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])   # 이진분류에 한해 로스함수는 무조건 99퍼센트로 'binary_crossentropy'
                                      # 컴파일에있는 metrics는 평가지표라고도 읽힘

from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import datetime
date = datetime.datetime.now()
date = date.strftime("%m%d_%H%M") # 0707_1723
print(date)

filepath = './_ModelCheckPoint/' + current_name + '/'
filename = '{epoch:04d}-{val_loss:.4f}.hdf5'

earlyStopping = EarlyStopping(monitor='val_loss', patience=100, mode='auto', verbose=1, 
                              restore_best_weights=True)        

mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, 
                      filepath= "".join([filepath, date, '_', filename])
                      )

hist = model.fit(x_train, y_train, epochs=100, batch_size=10,
                 validation_split=0.2,
                 callbacks=[earlyStopping, mcp],
                 verbose=1)

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)

print(y_predict)
y_predict = y_predict.round(0)
print(y_predict)


# y_summit = model.predict(test_set)

# print(y_summit)
# print(y_summit.shape) # (418, 1)
# y_summit = y_summit.round()
# df = pd.DataFrame(y_summit)
# print(df)
# oh = OneHotEncoder(sparse=False) # sparse=true 는 매트릭스반환 False는 array 반환
# y_summit = oh.fit_transform(df)
# print(y_summit)
# y_summit = np.argmax(y_summit, axis= 1)
# submission_set = pd.read_csv(path + 'gender_submission.csv', # + 명령어는 문자를 앞문자와 더해줌
#                              index_col=0) # index_col=n n번째 컬럼을 인덱스로 인식

# print(submission_set)

# submission_set['Survived'] = y_summit
# print(submission_set)


# submission_set.to_csv(path + 'submission_maxabs.csv', index = True)

acc= accuracy_score(y_test, y_predict)
print('loss : ' , loss)
print('acc스코어 : ', acc) 


#drop out 이전

#drop out 이후