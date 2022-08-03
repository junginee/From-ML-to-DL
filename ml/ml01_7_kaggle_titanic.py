
import numpy as np
import pandas as pd 

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import MaxAbsScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score, mean_squared_error



#1.데이터

path = './_data/kaggle_titanic/' # ".은 현재 폴더"
train_set = pd.read_csv(path + 'train.csv',
                        index_col=0)
test_set = pd.read_csv(path + 'test.csv', #예측에서 쓸거야!!
                       index_col=0)

# print(test_set) 
# print(train_set.isnull().sum()) 

train_set = train_set.fillna(train_set.median())

# print(test_set.isnull().sum())


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

from tqdm import tqdm_notebook
for col in tqdm_notebook(cols):
    le = LabelEncoder()
    train_set[col]=le.fit_transform(train_set[col])
    test_set[col]=le.fit_transform(test_set[col])
    
x = train_set.drop(['Survived'],axis=1) #axis는 컬럼 
print(x) #(891, 9)

y = train_set['Survived']
print(y.shape) #(891,)

gender_submission = pd.read_csv(path + 'gender_submission.csv',#예측에서 쓸거야!!
                       index_col=0)


x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.91,shuffle=True ,random_state=100)

scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train) 
x_test = scaler.transform(x_test)
test_set = scaler.transform(test_set)
print(np.min(x_train))  # 0.0
print(np.max(x_train))  # 1.0
print(np.min(x_test))  # 1.0
print(np.max(x_test))  # 1.0


#2. 모델 구성
model = LinearSVC() 



#3. 컴파일,훈련
model.fit(x_train, y_train)



#4. 평가, 예측
print("캐글타이타닉")
results = model.score(x_test, y_test)

print("acc : ", round(results,3)) 

# from sklearn.metrics import r2_score, accuracy_score  
# y_predict = model.predict(x_test)

# acc= accuracy_score(y_test, y_predict)
# print('acc스코어 : ', round(acc,3)) 

# 캐글타이타닉
# acc :  0.753
