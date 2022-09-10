import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler,MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.svm import SVC
from tensorflow.python.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score,  f1_score
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings("ignore")



#1.데이터

path = './_data/kaggle_titanic/'
train_set = pd.read_csv(path + 'train.csv',
                        index_col=0)
test_set = pd.read_csv(path + 'test.csv',
                       index_col=0)
train_set = train_set.fillna(train_set.median())

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

gender_submission = pd.read_csv(path + 'gender_submission.csv',
                       index_col=0)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.91,shuffle=True ,random_state=100)


#2. 모델
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC


scaler = StandardScaler()
# scaler = MinMaxScaler()
# scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = SVC(), LogisticRegression(), DecisionTreeClassifier(), LogisticRegression(), 
       # XGBClassifier()

for i in model :
    model = i
    new_model = BaggingClassifier(i,
                          n_estimators=200, 
                          n_jobs=-1,
                          random_state=123
                          )
    model.fit(x_train, y_train)
    print(i,'모델 score :',round(model.score(x_test, y_test),4),'\n') 


# SVC() 모델 score : 0.8025 

# LogisticRegression() 모델 score : 0.7654

# DecisionTreeClassifier() 모델 score : 0.8272 

# LogisticRegression() 모델 score : 0.7654    
