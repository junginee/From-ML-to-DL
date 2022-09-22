from torchvision.datasets import MNIST
from torch.utils.data import TensorDataset, DataLoader  #텐서 데이처로 변환, 배치적용된 데이터 블러오기
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')  
print('torch : ', torch.__version__,'사용DEVICE : ', DEVICE)

import torchvision.transforms as tr
trasnf = tr.Compose([tr.Resize(15), tr.ToTensor()])

#1. 데이터
path = 'D:\study\_data\\torch_data\mnist/'

train_dataset = MNIST(path, train=True, download=True)
test_dataset = MNIST(path, train=False, download=True)

# print(train_dataset[0][0].shape)

x_train, y_train = train_dataset.data/255.  , train_dataset.targets
x_test, y_test = test_dataset.data/255.  , test_dataset.targets

print(x_train.shape, x_test.size())     #torch.Size([60000, 28, 28]) torch.Size([10000, 28, 28])
print(y_train.shape, y_test.size())     #torch.Size([60000]) torch.Size([10000])
print(np.min(x_train.numpy()), np.max(x_train.numpy()))       #0.0 1.0
print(np.min(x_test.numpy()), np.max(x_test.numpy()))         #0.0 1.0

'''
cf. keras 넘파이 배열 전환
to_numpy(), values()
'''

x_train, x_test = x_train.view(-1, 784), x_test.view(-1, 28*28)
print(x_train.shape, x_test.size())     #torch.Size([60000, 784]) torch.Size([10000, 784])

'''
x_train, x_test = x_train.reshape(-1, 784), x_test.reshape(-1, 28*28)
print(x_train.shape, x_test.size())
'''

train_dset = TensorDataset(x_train, y_train)
test_dset = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_dset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dset, batch_size=32, shuffle=False)

#2. 모델
class DNN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        
        self.hidden_layer1 = nn.Sequential(
            nn.Linear(num_features, 100),
            nn.ReLU()
        )    
        self.hidden_layer2 = nn.Sequential(
        nn.Linear(100, 100),
        nn.ReLU()
        )        
        self.hidden_layer3 = nn.Sequential(
        nn.Linear(100, 100),
        nn.ReLU()
        )
        self.hidden_layer4 = nn.Sequential(
        nn.Linear(100, 100),
        nn.ReLU()            
        )
        self.output_layer = nn.Linear(100,10)
        
    def forward(self, x):
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.hidden_layer3(x)
        x = self.hidden_layer4(x)    
        x = self.output_layer(x)
        return x
    
model = DNN(784).to(DEVICE)    

#3. 컴파일, 훈련
criterion = nn.CrossEntropyLoss() #sparsecategorical entropy와 유사한 작용

optimizer = optim.Adam(model.parameters(), lr=1e-4)

def train(model, criterion, optimizer, loader):
    
    epoch_loss = 0
    epoch_acc = 0
    
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
        
        optimizer.zero_grad()

        hypothesis = model(x_batch)

        loss = criterion(hypothesis, y_batch)
        loss.backward() 
        optimizer.step()       

        epoch_loss += loss.item()
        
        y_predict = torch.argmax(hypothesis, 1)
        acc = (y_predict == y_batch).float().mean()
        
        epoch_acc += acc.item()
    
    return epoch_loss / len(loader), epoch_acc / len(loader)   
# hist = model.fit(x_train, y_train)         #hist에는 loss와 acc가 들어감
# 엄밀히 얘기하면 hist라고 정의하기보다, loss와 acc를 반환

def evaluate(model, criterion, loader):
    model.eval()
    
    epoch_loss = 0
    epoch_acc = 0
    
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            
            hypothesis = model(x_batch)
            
            loss = criterion(hypothesis, y_batch)
            
            epoch_loss += loss.item()
            
            y_predict = torch.argmax(hypothesis, 1)
            acc = (y_predict == y_batch).float().mean()
            
            epoch_acc += acc.item()
        
        return epoch_loss / len(loader), epoch_acc / len(loader)        
    
# loss, acc = model.evaluate(x_test, y_test)

epochs = 20 
for epoch in range(1, epochs + 1) :

    loss, acc = train(model, criterion, optimizer, train_loader)  

    val_loss, val_acc = evaluate(model, criterion, test_loader)
    
    print('epoch:{}, loss:{:.4f}, acc:{:.3f}, val_loss:{:.4f}, val_acc:{:.3f}'.format(
        epoch, loss, acc, val_loss, val_acc
    ))