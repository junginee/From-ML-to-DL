# [실습]
# pandas의  y 라벨의 종류가 무엇인지 확인하는 함수 쓸것
# numpy 에서는 np.unique(y,return_counts=True)

from pydoc import describe
import numpy as np
import pandas as pd 

from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm_notebook #문자열을 숫자로 전환
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score, mean_squared_error



#1.데이터
path = './_data/kaggle_titanic/' # ".은 현재 폴더"
train_set = pd.read_csv(path + 'train.csv',
                        index_col=0)
test_set = pd.read_csv(path + 'test.csv', #예측에서 쓸거야!!
                       index_col=0)
'''
print(train_set) # [891 rows x 11 columns]

print("===============================")

print(train_set.describe())
#          Survived      Pclass         Age       SibSp       Parch        Fare
# count  891.000000  891.000000  714.000000  891.000000  891.000000  891.000000
# mean     0.383838    2.308642   29.699118    0.523008    0.381594   32.204208
# std      0.486592    0.836071   14.526497    1.102743    0.806057   49.693429
# min      0.000000    1.000000    0.420000    0.000000    0.000000    0.000000
# 25%      0.000000    2.000000   20.125000    0.000000    0.000000    7.910400
# 50%      0.000000    3.000000   28.000000    0.000000    0.000000   14.454200
# 75%      1.000000    3.000000   38.000000    1.000000    0.000000   31.000000
# max      1.000000    3.000000   80.000000    8.000000    6.000000  512.329200

print("===============================")
print(train_set.info())

#  #   Column    Non-Null Count  Dtype
# ---  ------    --------------  -----
#  0   Survived  891 non-null    int64
#  1   Pclass    891 non-null    int64
#  2   Name      891 non-null    object
#  3   Sex       891 non-null    object
#  4   Age       714 non-null    float64
#  5   SibSp     891 non-null    int64
#  6   Parch     891 non-null    int64
#  7   Ticket    891 non-null    object
#  8   Fare      891 non-null    float64
#  9   Cabin     204 non-null    object
#  10  Embarked  889 non-null    object
'''

print(test_set) # [418 rows x 10 columns]
print(train_set.isnull().sum()) #각 컬럼당 결측치의 합계
# Survived      0
# Pclass        0
# Name          0
# Sex           0
# Age         177
# SibSp         0
# Parch         0
# Ticket        0
# Fare          0
# Cabin       687
# Embarked      2

train_set = train_set.fillna(train_set.median())

print(test_set.isnull().sum())
# Pclass        0
# Name          0
# Sex           0
# Age          86
# SibSp         0
# Parch         0
# Ticket        0
# Fare          1
# Cabin       327
# Embarked      0

drop_cols = ['Cabin']
train_set.drop(drop_cols, axis = 1, inplace =True)
test_set = test_set.fillna(test_set.mean())
train_set['Embarked'].fillna('S')
train_set = train_set.fillna(train_set.mean())
print("===============================")
print(train_set) 
print(train_set.isnull().sum())

test_set.drop(drop_cols, axis = 1, inplace =True)
cols = ['Name','Sex','Ticket','Embarked']

for col in tqdm_notebook(cols):
    le = LabelEncoder()
    train_set[col]=le.fit_transform(train_set[col])
    test_set[col]=le.fit_transform(test_set[col])
    
x = train_set.drop(['Survived'],axis=1) #axis는 컬럼 
print(x) #(891, 9)
y = train_set['Survived']
print(y.shape) #(891,)


# test_set.drop(drop_cols, axis = 1, inplace =True)
gender_submission = pd.read_csv(path + 'gender_submission.csv',#예측에서 쓸거야!!
                       index_col=0)
# y의 라벨값 : (array([0, 1], dtype=int64), array([549, 342], dtype=int64))

###########(pandas 버전 원핫인코딩)###############
# y_class = pd.get_dummies((y))
# print(y_class.shape) # (891, 2)

# 해당 기능을 통해 y값을 클래스 수에 맞는 열로 늘리는 원핫 인코딩 처리를 한다.
#1개의 컬럼으로 [0,1,2] 였던 값을 ([1,0,0],[0,1,0],[0,0,1]과 같은 shape로 만들어줌)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.91,shuffle=True ,random_state=100)


#2. 모델 구성
model = Sequential()
model.add(Dense(100,input_dim=9))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일,훈련
earlyStopping = EarlyStopping(monitor='loss', patience=200, mode='min', 
                              verbose=1,restore_best_weights=True)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, batch_size=20, 
                validation_split=0.3,
                callbacks = [earlyStopping],
                verbose=2
                )
#다중 분류 모델은 'categorical_crossentropy'만 사용한다 !!!!

#4.  평가,예측
# loss,acc = model.evaluate(x_test,y_test)
# print('loss :',loss)
# print('accuracy :',acc)
# print("+++++++++  y_test       +++++++++")
# print(y_test[:5])
# print("+++++++++  y_pred     +++++++++++++")
# result = model.evaluate(x_test,y_test) 위에와 같은 개념 [0] 또는 [1]을 통해 출력가능
# print('loss :',result[0])
# print('accuracy :',result[1])

y_predict = model.predict(x_test)
y_predict[(y_predict<0.5)] = 0  
y_predict[(y_predict>=0.5)] = 1  
print(y_predict) 
print(y_test.shape) #(134,)

acc = accuracy_score(y_test, y_predict)
print('acc 스코어 :', acc)

y_summit = model.predict(test_set)
gender_submission['Survived'] = y_summit
submission = gender_submission.fillna(gender_submission.mean())
submission [(submission <0.5)] = 0  
submission [(submission >=0.5)] = 1  
submission = submission.astype(int)
submission.to_csv('test21.csv',index=True)


# acc 스코어 : 0.7654320987654321

