import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용 DEVICE : ', DEVICE)
# torch :  1.12.1 사용 DEVICE :  cuda:0

#1. 데이터
path = './_data/bike/'
train_set = pd.read_csv(path + 'train.csv')
test_set = pd.read_csv(path + 'test.csv')

#### 결측치 처리 ####
train_set = train_set[train_set['weather'] != 4]  # weather가 4인 데이터는 이상치(폭우, 폭설이 내리는 날 저녁 6시에 대여)

train_set['datetime'] = pd.to_datetime(train_set['datetime'])
test_set['datetime'] = pd.to_datetime(test_set['datetime'])

# 날짜 feature 생성
L = ['year', 'month', 'date', 'hour', 'weekday']
train_set = train_set.join(pd.concat([getattr(train_set['datetime'].dt, i).rename(i) for i in L], axis=1))
test_set = test_set.join(pd.concat([getattr(test_set['datetime'].dt, i).rename(i) for i in L], axis=1))

# drop_features
drop_features = ['casual', 'registered', 'datetime', 'year', 'date', 'windspeed', 'month']
train_set = train_set.drop(drop_features, axis=1)
test_set = test_set.drop(['datetime', 'year', 'date', 'windspeed', 'month'], axis=1)

x = train_set.drop(['count'], axis=1)
y = train_set['count']

x = torch.FloatTensor(x.to_numpy())
y = torch.FloatTensor(y.to_numpy())

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.9, shuffle=True, random_state=13
)
x_train = torch.FloatTensor(x_train)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
x_test = torch.FloatTensor(x_test)
y_test = torch.FloatTensor(y_test).unsqueeze(-1).to(DEVICE)

from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)

print(x_train.size())   # torch.Size([9796, 9])


############################## DataLoader 시작 #################################
from torch.utils.data import TensorDataset, DataLoader
train_set = TensorDataset(x_train, y_train)
test_set = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
test_loader = DataLoader(test_set, batch_size=128, shuffle=True)


#2. 모델 
# model = nn.Sequential(
#     nn.Linear(9, 32),
#     nn.SELU(),
#     nn.Linear(32, 64),
#     nn.ReLU(),
#     nn.Linear(64, 128),
#     nn.ReLU(),
#     nn.Linear(128, 64),
#     nn.ReLU(),
#     nn.Linear(64, 32),
#     nn.ReLU(),
#     nn.Linear(32, 1),
# ).to(DEVICE) 
class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 32)
        self.linear2 = nn.Linear(32, 64)
        self.linear3 = nn.Linear(64, 128)
        self.linear4 = nn.Linear(128, 64)
        self.linear5 = nn.Linear(64, 32)
        self.linear6 = nn.Linear(32, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, input_size):
        x = self.linear1(input_size)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        x = self.relu(x)
        x = self.linear5(x)
        x = self.relu(x)
        x = self.linear6(x)
        return x

model = Model(9, 1).to(DEVICE)


#3. 컴파일, 훈련
criterion = nn.MSELoss()

optimizer = optim.Adam(model.parameters(), lr=0.05, weight_decay=1e-7)

def train(model, criterion, optimizer, loader):
    # model.train()     # 디폴트
    total_loss = 0
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        hypothesis = model(x_batch)
        loss = criterion(hypothesis, y_batch)
        loss.backward()     # 역전파
        optimizer.step()    # 가중치 갱신
        total_loss += loss.item()
        
    return total_loss / len(loader)

EPOCHS = 100
for epoch in range(1, EPOCHS+1):
    running_loss =0.0
    loss = train(model, criterion, optimizer, x_train, y_train)
    if epoch % 10 == 0:
        print('epoch : {}, loss : {:.8f}'.format(epoch, loss))

#4. 평가, 예측
print("========================== 평가, 예측 =============================")
def evaluate(model, criterion, loader):
    model.eval()
    total_loss = 0
    for x_batch, y_batch in loader:
        with torch.no_grad():
            hypothesis = model(x_batch)
            loss = criterion(hypothesis, y_batch)
            total_loss += loss.item()
        return total_loss
    
loss2 = evaluate(model, criterion, test_loader)
print('loss : ', loss2)

y_predict = model(x_test)
print(y_predict[:10])

from sklearn.metrics import r2_score
score = r2_score(y_test.detach().cpu().numpy(), 
                 y_predict.detach().cpu().numpy())
print('r2_score : ', score)

# ========================== 평가, 예측 =============================
# loss :  8184.416015625
# r2_score :  0.7492767949068668