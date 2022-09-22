import numpy as np
import pandas as pd
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
path = 'D:\study_data\_data\kaggle_house/'
train_set = pd.read_csv(path + 'train.csv')
test_set = pd.read_csv(path + 'test.csv')
train_set.set_index('Id', inplace=True)
test_set.set_index('Id', inplace=True)
test_id_index = train_set.index
trainLabel = train_set['SalePrice']
train_set.drop(['SalePrice'], axis=1, inplace=True)
############################################
################### 트레인,테스트 합치기 ##################
alldata = pd.concat((train_set, test_set), axis=0)
alldata_index = alldata.index
################## NA 값 20프로 이상은 drop! ##########
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

x = train_set
y = trainLabel
import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F
 
USE_CUDA = torch.cuda.is_available
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용DEVICE : ', DEVICE)

x = x.to_numpy()
y = y.to_numpy()

x = torch.FloatTensor(x)
y = torch.FloatTensor(y)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.8,shuffle=True, random_state=123)

x_train = torch.FloatTensor(x_train)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
x_test = torch.FloatTensor(x_test)
y_test = torch.FloatTensor(y_test).unsqueeze(1).to(DEVICE)

print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)

#2.모델 
# model = nn.Sequential(
#     nn.Linear(74,64),
#     nn.ReLU(),
#     nn.Linear(64,32),
#     nn.ReLU(),
#     nn.Linear(32,16),
#     nn.ReLU(),
#     nn.Linear(16,1),
      
# ).to(DEVICE)


class Model(nn.Module):
    def __init__(self,input_dim,output_dim):
        super().__init__()
        self.L1 = nn.Linear(input_dim, 64)
        self.L2 = nn.Linear(64, 32)
        self.L3 = nn.Linear(32, 16)
        self.L4 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        
        
    def forward(self,input_size):
        x = self.L1(input_size)
        x = self.relu(x)
        x = self.L2(x)
        x = self.relu(x)
        x = self.L3(x)
        x = self.relu(x)
        x = self.L4(x)
        return x 

model = Model(74,1).to(DEVICE)

#3.컴파일,훈련 
criterion = nn.MSELoss()  
optimizer = optim.Adam(model.parameters(),lr=0.01)

def train(model,criterion,optimizer,x_train,y_train):
    optimizer.zero_grad()
    
    hypothesis = model(x_train)
    
    loss = criterion(hypothesis,y_train)
    
    loss.backward()
    optimizer.step()
    return loss.item()

epochs = 2000
for epoch in range(epochs+1):
    loss = train(model,criterion,optimizer,x_train,y_train)
    print('epochs : {}, loss : {}'.format(epochs,loss))
    
#4.평가,예측
def evaluate(model,criterion,x_test,y_test):
    model.eval()
    
    with torch.no_grad():
        y_predict = model(x_test)
        results = criterion(y_predict,y_test)    
    return results.item()

loss2 = evaluate(model,criterion,x_test,y_test)
print('최종 loss : ', loss2)
from sklearn.metrics import accuracy_score,r2_score
y_predict = model(x_test)
score = r2_score(y_predict.cpu().detach(),y_test.cpu().detach())
print('r2 : ', score)

# 최종 loss :  1283797248.0
# r2 :  0.7919833312796748
