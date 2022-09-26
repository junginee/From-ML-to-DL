import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.layers import Dense,Input,Dropout
from keras.layers.recurrent import LSTM, SimpleRNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import datetime as dt
from sklearn.preprocessing import MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler
from tensorflow.python.keras.callbacks  import EarlyStopping,ModelCheckpoint
import datetime
     
#1. 데이터
path = './_data/kaggle_bike/'
train_set = pd.read_csv(path + 'train.csv') 
            
test_set = pd.read_csv(path + 'test.csv') 

train_set["hour"] = [t.hour for t in pd.DatetimeIndex(train_set.datetime)]
train_set["day"] = [t.dayofweek for t in pd.DatetimeIndex(train_set.datetime)]
train_set["month"] = [t.month for t in pd.DatetimeIndex(train_set.datetime)]
train_set['year'] = [t.year for t in pd.DatetimeIndex(train_set.datetime)]
train_set['year'] = train_set['year'].map({2011:0, 2012:1})

test_set["hour"] = [t.hour for t in pd.DatetimeIndex(test_set.datetime)]
test_set["day"] = [t.dayofweek for t in pd.DatetimeIndex(test_set.datetime)]
test_set["month"] = [t.month for t in pd.DatetimeIndex(test_set.datetime)]
test_set['year'] = [t.year for t in pd.DatetimeIndex(test_set.datetime)]
test_set['year'] = test_set['year'].map({2011:0, 2012:1})

train_set.drop('datetime',axis=1,inplace=True) 
train_set.drop('casual',axis=1,inplace=True)
train_set.drop('registered',axis=1,inplace=True)

test_set.drop('datetime',axis=1,inplace=True) 


x = train_set.drop(['count'], axis=1)  
print(x)
print(x.columns)
print(x.shape) # (10886, 12)

y = train_set['count'] 
print(y)
print(y.shape) # (10886,)

x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.75,random_state=31)   
      

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) 
test_set = scaler.transform(test_set)      
                                                    

#2. 모델구성
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=12))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))

date=datetime.datetime.now()
date=date.strftime('%m%d_%H%M')

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
     

earlyStopping =EarlyStopping(monitor = 'val_loss',patience=30,mode='min',restore_best_weights=True,verbose=1)

# filepath='./_ModelCheckpoint/k24/'
# filename='{epoch:04d}-{val_loss:.4f}.hdf5'

# mcp = ModelCheckpoint(monitor = 'val_loss',mode = 'auto', save_best_only=True, verbose=1,
#                       filepath="".join([filepath,'kaggle_bike',date,'_',filename]))

model.fit(x_train, y_train, epochs=800, batch_size=60,validation_split=0.2,callbacks=[earlyStopping], verbose=2)

#4. 평가, 예측
loss = model.evaluate(x, y) 
print('loss : ', loss)

y_predict = model.predict(x_test)


def RMSE(a, b): 
    return np.sqrt(mean_squared_error(a, b))

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)


#================================ dorpout 적용 ================================#
# loss :  2176.55322265625
# RMSE :  53.909264740796814
