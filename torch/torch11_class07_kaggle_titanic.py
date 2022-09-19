import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용 DEVICE : ', DEVICE)

#1. 데이터
path = './_data/kaggle_titanic/'
train_set = pd.read_csv(path + 'train.csv', index_col=0)
test_set = pd.read_csv(path + 'test.csv', index_col=0)

# train_tensor = torch.tensor(train_set.to_numpy())
# test_tensor = torch.tensor(test_set.to_numpy())

# train_set = torch.FloatTensor(train_set)
# test_set = torch.FloatTensor(test_set)

# 데이터 전처리
train_set[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_set[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_set[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_set[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# Ticket, Cabin, Name 삭제
train_set = train_set.drop(['Ticket', 'Cabin'], axis=1)
test_set = test_set.drop(['Ticket', 'Cabin'], axis=1)

train_set = train_set.drop(['Name'], axis=1)
test_set = test_set.drop(['Name'], axis=1)

# Age NaN값 변환
train_set['Age'] = train_set['Age'].fillna(train_set.Age.dropna().mode()[0])
test_set['Age'] = test_set['Age'].fillna(train_set.Age.dropna().mode()[0])

# Embarked, Sex NaN값 및 Object => int 변환
train_set['Embarked'] = train_set['Embarked'].fillna(train_set.Embarked.dropna().mode()[0])
train_set['Embarked'] = train_set['Embarked'].map({'S':0, 'C':1, 'Q':2}).astype(int)

test_set['Embarked'] = test_set['Embarked'].fillna(test_set.Embarked.dropna().mode()[0])
test_set['Embarked'] = test_set['Embarked'].map({'S':0, 'C':1, 'Q':2}).astype(int)

train_set['Sex'] = train_set['Sex'].fillna(train_set.Sex.dropna().mode()[0])
train_set['Sex'] = train_set['Sex'].map({'male':0, 'female':1}).astype(int)

test_set['Sex'] = test_set['Sex'].fillna(test_set.Sex.dropna().mode()[0])
test_set['Sex'] = test_set['Sex'].map({'male':0, 'female':1}).astype(int)

# x, y 데이터
x = train_set.drop(['Survived'], axis=1)
print(x)
print(x.columns)    # 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'

y = train_set['Survived']
print(y)
print(y.shape)  # (891,)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.9, shuffle=True, random_state=1234
)

x_train = torch.FloatTensor(x_train.to_numpy()).to(DEVICE) # dataframe을 numpy로 변환
y_train = torch.FloatTensor(y_train.to_numpy()).unsqueeze(1).to(DEVICE)
x_test = torch.FloatTensor(x_test.to_numpy()).to(DEVICE)
y_test = torch.FloatTensor(y_test.to_numpy()).unsqueeze(-1).to(DEVICE)

print(x_train.size())   # torch.Size([801, 7])

############################## DataLoader 시작 #################################
from torch.utils.data import TensorDataset, DataLoader
train_set = TensorDataset(x_train, y_train)
test_set = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32, shuffle=True)


#2. 모델
# model = nn.Sequential(
#     nn.Linear(7, 64),
#     nn.ELU(),
#     nn.Linear(64, 128),
#     nn.ELU(),
#     nn.Linear(128, 64),
#     nn.ReLU(),
#     nn.Linear(64, 32),
#     nn.ReLU(),
#     nn.Linear(32, 1),
#     nn.Sigmoid()
# ).to(DEVICE)
class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64, 128)
        self.linear3 = nn.Linear(128, 64)
        self.linear4 = nn.Linear(64, 32)
        self.linear5 = nn.Linear(32, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
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
        x = self.sigmoid(x)
        return x
    
model = Model(7, 1).to(DEVICE)


#3. 컴파일, 훈련
criterion = nn.BCELoss()

optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, loader):
    # model.train()
    total_loss = 0
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        hypothesis = model(x_batch)
        loss = criterion(hypothesis, y_batch)       
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

EPOCHS = 100
for epoch in range(1, EPOCHS+1):
    loss = train(model, criterion, optimizer, train_loader)
    if epoch % 10 == 0:
        print('epoch : {}, loss : {:.8f}'.format(epoch, loss))
    
#4. 평가, 예측
print("=============================== 평가, 예측 =================================")
def evaluate(model, criterion, loader):
    model.eval()
    total_loss = 0
    for x_batch, y_batch in loader:
        with torch.no_grad():
            hypothesis = model(x_batch)
            loss = criterion(hypothesis, y_batch)
        return loss.item()

loss2 = evaluate(model, criterion,test_loader)
print('loss : ', loss2)

y_predict = (model(x_test) >= 0.5).float()
print(y_predict[:10])

score = (y_predict == y_test).float().mean()
print('accuracy : {:.4f}'.format(score))

from sklearn.metrics import accuracy_score
score = accuracy_score(y_test.cpu(), y_predict.cpu())
print('accuracy_score : ', score)

# =============================== 평가, 예측 =================================
# loss :  0.36176639795303345
# accuracy : 0.8556
# accuracy_score :  0.8555555555555555