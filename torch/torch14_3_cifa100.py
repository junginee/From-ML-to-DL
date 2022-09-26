from torchvision.datasets import CIFAR100
import torch 
from torch.utils.data import TensorDataset,DataLoader
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F
import numpy as np 

USE_CUDA = torch.cuda.is_available
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')

# import torchvision.transforms as tr
# transf = tr.Compose([tr.Resize(32),tr.ToTensor()])

#1. 데이터
path = 'D:\study_data\_data/torch_data/'

# train_datasets = CIFAR10(path, train= True, download=True, transform= transf)
# test_datasets = CIFAR10(path, train= False, download=True, transform= transf)

# print(train_datasets[0][0].shape)  #torch.Size([1, 15, 15])
train_datasets = CIFAR100(path, train= True, download=True)
test_datasets = CIFAR100(path, train= False, download=True)

x_train,y_train = train_datasets.data/255. , train_datasets.targets
x_test,y_test = test_datasets.data/255. , test_datasets.targets

x_train = torch.FloatTensor(x_train)
y_train = torch.IntTensor(y_train)
x_test = torch.FloatTensor(x_test)
y_test = torch.LongTensor(y_test)

print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)


x_train,x_test = x_train.view(50000, 32*32*3), x_test.reshape(10000, 32*32*3) #  == reshape

train_set = TensorDataset(x_train,y_train)
test_set = TensorDataset(x_test,y_test)

train_loader = DataLoader(train_set,batch_size=32,shuffle=True)
test_loader = DataLoader(test_set,batch_size=32,shuffle=False)

#2.모델
class DNN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        
        self.hidden_layer1 = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Dropout(0.1))
        self.hidden_layer2 = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1)) 
        self.hidden_layer3 = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.1))
        self.hidden_layer4 = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(0.5),
            nn.Dropout(0.2))
        self.hidden_layer5 = nn.Sequential(
            nn.Linear(16, 10),
            nn.ReLU(0.5),
            nn.Dropout(0.1))   
        self.output_layer = nn.Linear(10,100)

    def forward(self, x):
        x = self.hidden_layer1(x)
        x = self.hidden_layer2(x)
        x = self.hidden_layer3(x)
        x = self.hidden_layer4(x)
        x = self.hidden_layer5(x)
        x = self.output_layer(x)
        return x

model = DNN(32*32*3).to(DEVICE)

#3. 컴파일,훈련
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=1e-4)

def train(model,criterion,optimizer,loader):
    
    epoch_loss = 0
    epoch_acc = 0
    for x_batch,y_batch in loader:
        x_batch,y_batch = x_batch.to(DEVICE),y_batch.to(DEVICE)
        
        optimizer.zero_grad()
        hypothesis = model(x_batch)
        loss = criterion(hypothesis, y_batch)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        y_predict = torch.argmax(hypothesis, 1)
        acc = (y_predict == y_batch).float().mean() # bool 로 나온것을 float으로
        epoch_acc += acc.item()
    return epoch_loss / len(loader), epoch_acc / len(loader)         


def evaluate(model,criterion,loader):
    model.eval() 
    epoch_loss = 0
    epoch_acc = 0 

    with torch.no_grad():
    
        for x_batch,y_batch in loader:
            x_batch,y_batch = x_batch.to(DEVICE),y_batch.to(DEVICE)
 
            hypothesis = model(x_batch)
            loss = criterion(hypothesis,y_batch)
            epoch_loss += loss.item()
            y_predict = torch.argmax(hypothesis, 1)
            acc = (y_predict == y_batch).float().mean() # bool 로 나온것을 float으로
            epoch_acc += acc.item()
    return epoch_loss / len(loader), epoch_acc / len(loader)


epochs = 20
for epoch in range(epochs+1):
    loss,acc = train(model,criterion,optimizer,train_loader)

    val_loss, val_acc = evaluate(model,criterion, test_loader)

    print('epochs : {}, loss : {:.4f}, acc : {:.3f}, val_loss : {:.4f}, val_acc : {:.3f}'.format(epochs, loss, acc, val_loss, val_acc))
