from calendar import EPOCH
from sklearn.datasets import load_breast_cancer, load_digits, load_wine, fetch_covtype
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용 DEVICE : ', DEVICE)
# torch :  1.12.1 사용 DEVICE :  cuda:0


#1. 데이터 
datasets = fetch_covtype()
x = datasets.data
y = datasets['target']

x = torch.FloatTensor(x)
y = torch.LongTensor(y)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, shuffle=True, random_state=123, stratify=y)

x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train).to(DEVICE)
x_test = torch.FloatTensor(x_test)
y_test = torch.LongTensor(y_test).to(DEVICE)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)

print(x_train.size(), y_train.size())
print(x_test.shape, y_test.shape)

# torch.Size([406708, 54]) torch.Size([406708])
# torch.Size([174304, 54]) torch.Size([174304])

#2. 모델

# model = nn.Sequential(
#     nn.Linear(54, 64),
#     nn.ReLU(),
#     nn.Linear(64, 32),
#     nn.ReLU(),
#     nn.Linear(32, 48),
#     nn.ReLU(),
#     nn.Linear(48, 16),
#     nn.ReLU(),
#     nn.Linear(16, 32),
#     nn.ReLU(),
#     nn.Linear(32, 16),
#     nn.ReLU(),
#     nn.Linear(16, 7),
#     # nn.Sigmoid(),
# ).to(DEVICE)

class Model(nn.Module):
    def __init__(self, input_dim, output_dim ):
        super().__init__()
        self.Linear1 = nn.Linear(input_dim, 64)
        self.Linear2 = nn.Linear(64, 32)
        self.Linear3 = nn.Linear(32, 48)
        self.Linear4 = nn.Linear(48, 16)
        self.Linear5 = nn.Linear(16, 32)
        self.Linear6 = nn.Linear(32, 16)
        self.Linear7 = nn.Linear(16, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_size):
        x = self.Linear1(input_size)
        x = self.relu(x) 
        x = self.Linear2(x)
        x = self.relu(x)    
        x = self.Linear3(x)
        x = self.relu(x)    
        x = self.Linear4(x)
        x = self.relu(x)   
        x = self.Linear5(x)
        x = self.relu(x)   
        x = self.Linear6(x)
        x = self.relu(x)
        x = self.Linear7(x)
        x = self.relu(x)
        return x
model = Model(54,7).to(DEVICE)       
   


#3. 컴파일, 훈련

criterion = nn.CrossEntropyLoss() # 이진분류
optimizer = optim.Adam(model.parameters(), lr=0.01)


def train(model, criterion, optimizer, x_train, y_train):
    # model.trian()
    optimizer.zero_grad()
    hypothesis = model(x_train)
    loss = criterion(hypothesis, y_train)
    loss.backward()
    optimizer.step()
    return loss.item()

EPOCHS = 1500

for epoch in range(1, EPOCHS+1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    print('epoch : {}, loss :{}'.format(epoch, loss))    

def evaluate(model, criterion, x_test, y_test):
    model.eval()
    
    with torch.no_grad():
        hypothesis = model(x_test)
        loss = criterion(hypothesis, y_test)
        return loss.item()
    
loss = evaluate(model, criterion, x_test, y_test)
print('최종 LOSS : ', loss)

# y_pred = model(x_test)
# print(y_pred[:10])

y_pred = (model(x_test) >= 0.5).float()
print(y_pred[:10])

score = (y_pred == y_test).float().mean()
print('ACC1 : {:.4f}'.format(score))

from sklearn.metrics import accuracy_score
# score = accuracy_score(y_test, y_pred)  # ERROR
# print('ACC : ', score)

score = accuracy_score(y_test.cpu(), y_pred.cpu())
print('ACC2 : ', score)