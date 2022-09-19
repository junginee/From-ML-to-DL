
from sklearn.datasets import load_diabetes
import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용DEVICE : ', DEVICE)

#1.데이터 
datasets = load_diabetes()
x,y = datasets.data,datasets.target

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


############################
from torch.utils.data import TensorDataset, DataLoader
train_set = TensorDataset(x_train, y_train)     #x, y를 합친다!
test_set = TensorDataset(x_test, y_test)         #x, y를 합친다!

print("============len(train_set)==============")
print(len(train_set))   #353


##########x, y 배치 합체#########
train_loader = DataLoader(train_set, batch_size=40, shuffle=True)
test_loader = DataLoader(test_set, batch_size=40, shuffle=True)


#2.모델 
class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.Linear1 = nn.Linear(input_dim, 64)
        self.Linear2 = nn.Linear(64, 32)
        self.Linear3 = nn.Linear(32,16)
        self.Linear4 = nn.Linear(16, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_size):
        x = self.Linear1(input_size)
        x = self.sigmoid(x)
        x = self.Linear2(x) 
        x = self.sigmoid(x)
        x = self.Linear3(x)
        x = self.relu(x)
        x = self.Linear4(x)
        return x
model = Model(10,1).to(DEVICE)     

#3.컴파일,훈련 
criterion = nn.MSELoss()  
optimizer = optim.Adam(model.parameters(),lr=0.01)

def train(model,criterion,optimizer,loader):
    total_loss = 0
    for x_batch, y_batch in loader:
        optimizer.zero_grad()
        hypothesis = model(x_batch)
        loss = criterion(hypothesis,y_batch)
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(loader)

epochs = 2000
for epoch in range(epochs+1):
    loss = train(model,criterion,optimizer, train_loader)
    if epoch % 10 == 0:
        print('epochs : {}, loss : {}'.format(epochs,loss))
    
#4.평가,예측
def evaluate(model,criterion,loader):
    model.eval()
    total_loss = 0
    
    for x_batch, y_batch in loader:
        with torch.no_grad():
            y_predict = model(x_batch)
            results = criterion(y_predict,y_batch)    
        return results.item()

loss2 = evaluate(model,criterion,test_loader)
print('최종 loss : ', loss2)

from sklearn.metrics import r2_score
y_predict = model(x_test)
score = r2_score(y_predict.cpu().detach(),y_test.cpu().detach())
# score = r2_score(y_predict.cpu(),y_test.cpu())
print('r2 : ', score)

'''
최종 loss :  6134.4609375
r2 :  -0.07198814200038872
'''
