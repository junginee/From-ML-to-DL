from torchvision.datasets import CIFAR10
from torch.utils.data import TensorDataset, DataLoader  
import torchvision.transforms as tr
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else ' cpu')
print('torch :', torch.__version__, '사용 DEVICE:', DEVICE)

#1. 데이터
path = './_data/torch_data/cifar10/'

train_dataset = CIFAR10(path, train=True, download=False)
test_dataset = CIFAR10(path, train=False, download=False)

x_train, y_train = train_dataset.data/255. , train_dataset.targets
x_test, y_test = test_dataset.data/255. , test_dataset.targets
print(np.unique(y_train))   #[0 1 2 3 4 5 6 7 8 9]

x_train = torch.FloatTensor(x_train).to(DEVICE)
y_train = torch.LongTensor(y_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)
y_test = torch.LongTensor(y_test).to(DEVICE)

print(x_train.shape, x_test.size())  
# torch.Size([50000, 32, 32, 3]) torch.Size([10000, 32, 32, 3])

x_train, x_test = x_train.reshape(50000, -1), x_test.reshape(10000, -1)
print(x_train.shape, x_test.size())  
# torch.Size([50000, 3072]) torch.Size([10000, 3072])

train_dset = TensorDataset(x_train, y_train)
test_dset = TensorDataset(x_test, y_test)

train_loader = DataLoader(train_dset, batch_size=32, shuffle = True)
test_loader = DataLoader(test_dset, batch_size=32, shuffle=False)


#2. 모델
class DNN(nn.Module):
    def __init__(self, num_features):
        super().__init__()

        self.hidden_layer1 = nn.Sequential(
            nn.Linear(num_features, 100),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.hidden_layer2 = nn.Sequential(
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Dropout(0.5)
        )    
        self.hidden_layer3 = nn.Sequential(
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.hidden_layer4 = nn.Sequential(
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Dropout(0.5)       
        )        
        self.output_layer = nn.Linear(100, 10)
    
    def forward(self, x):
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.hidden_layer3(x)
        x = self.hidden_layer4(x)
        x = self.output_layer(x)
        return x
model = DNN(3072).to(DEVICE)

#3. 컴파일, 훈련
criterion = nn.CrossEntropyLoss()

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
    
    #epoch:20, loss:1.4770, acc:0.470, val_loss:1.5017, val_acc:0.461
