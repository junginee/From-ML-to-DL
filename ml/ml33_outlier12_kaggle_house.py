import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error 
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,MinMaxScaler,StandardScaler,MaxAbsScaler,RobustScaler
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew
from sklearn.impute import SimpleImputer

#1.data 처리

############# id컬럼 index 처리 ##########
path = './_data/kaggle_house/'
train_set = pd.read_csv(path + 'train.csv')
test_set = pd.read_csv(path + 'test.csv')

train_set.set_index('Id', inplace=True)
test_set.set_index('Id', inplace=True)

test_id_index = train_set.index
trainLabel = train_set['SalePrice']
train_set.drop(['SalePrice'], axis=1, inplace=True)
############################################

###########이상치 처리######################
def dr_outlier(train_set):
    quartile_1 = train_set.quantile(0.25)
    quartile_3 = train_set.quantile(0.75)
    IQR = quartile_3 - quartile_1
    condition = (train_set < (quartile_1 - 1.5 * IQR)) | (train_set > (quartile_3 + 1.5 * IQR))
    condition = condition.any(axis=1)
    search_df = train_set[condition]

    return train_set, train_set.drop(train_set.index, axis=0)

dr_outlier(train_set)

############ 트레인,테스트 합치기 ############
alldata = pd.concat((train_set, test_set), axis=0)
alldata_index = alldata.index


############ NA 값 20프로 이상은 drop! ##########
NA_Ratio = 0.8 * len(alldata)
alldata.dropna(axis=1, thresh=NA_Ratio, inplace=True)

################### 수치형,카테고리형 분리,범위 설정 #############
alldata_obj = alldata.select_dtypes(include='object') 
alldata_num = alldata.select_dtypes(exclude='object')

for objList in alldata_obj:
    alldata_obj[objList] = LabelEncoder().fit_transform(alldata_obj[objList].astype(str))


##################### 소수 na 값 처리 ###################    
imputer = SimpleImputer(strategy='mean')
imputer.fit(alldata_num)
alldata_impute = imputer.transform(alldata_num)
alldata_num = pd.DataFrame(alldata_impute, columns=alldata_num.columns, index=alldata_index)  

###################### 분리한 데이터 다시 합치기 #####################
alldata = pd.merge(alldata_obj, alldata_num, left_index=True, right_index=True)  

##################### 트레인, 테스트 다시 나누기 ##################
train_set = alldata[:len(train_set)]
test_set = alldata[len(train_set):]

############### 트레인 데이터에 sale price 합치기 ##############
train_set['SalePrice'] = trainLabel

############### sale price 다시 드랍 #####################
train_set = train_set.drop(['SalePrice'], axis =1)
print(train_set)
print(trainLabel)
print(test_set)

###############################################################
x_train, x_test, y_train, y_test = train_test_split(train_set, trainLabel, train_size=0.8, 
                                                     random_state=58)

# minmax , standard 트레인테스트를 나눈뒤 해야한다. 과적합 방지와 같은 이치 
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = MaxAbsScaler()
scaler = RobustScaler()

scaler.fit(x_train)
x_train = scaler.transform(x_train) #스케일링한것을 보여준다.
x_test = scaler.transform(x_test) #test는 transfrom만 해야됨 
test_set = scaler.transform(test_set) # **최종테스트셋이 있는경우 여기도 스케일링을 적용해야함 **               
                                                    
#2.모델구성
model = Sequential()
model.add(Dense(100, input_dim=74))
model.add(Dense(100,activation ='relu'))
model.add(Dense(100,activation ='relu'))
model.add(Dense(100,activation ='relu'))
model.add(Dense(100,activation ='relu'))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss= 'mae', optimizer ='adam')
from tensorflow.python.keras.callbacks import EarlyStopping
earlyStopping= EarlyStopping(monitor='val_loss',patience=30,mode='min',restore_best_weights=True,verbose=1)
model.fit(x_train, y_train, epochs=2000, batch_size=50,validation_split=0.2,callbacks=earlyStopping, verbose=1)

# #4.평가,예측
loss = model.evaluate(train_set, trainLabel)
print('loss: ', loss)

y_predict = model.predict(x_test).flatten()

def RMSE(y_test, y_predict):
    return np.sqrt(mean_squared_error(y_test, y_predict)) #sqrt: 루트 씌우기

y_submmit = model.predict(test_set)

rmse = RMSE(y_test, y_predict)
print("RMSE : ", rmse)