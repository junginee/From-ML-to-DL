from sklearn.datasets import load_wine
import torch
import torch.nn as nn
import torch.optim as optim 

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')  
print('torch : ', torch.__version__,'사용DEVICE : ', DEVICE)

#. 1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets['target']

x = torch.FloatTensor(x)
y = torch.LongTensor(y)

print(x.shape, y.shape) #torch.Size([178, 13]) torch.Size([178]) 


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.7, shuffle = True, random_state=123, stratify=y)

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


print(x_train.shape) 
print(y_train.shape, y_test.shape)


#2.모델
# model = nn.Sequential(
#     nn.Linear(13,64),
#     nn.ReLU(),
#     nn.Linear(64,32),
#     nn.ReLU(),
#     nn.Linear(32,16),
#     nn.ReLU(),
#     nn.Linear(16,3),
#     nn.Sigmoid(),
   
# ).to(DEVICE)

class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.Linear1 = nn.Linear(input_dim, 64)
        self.Linear2 = nn.Linear(64, 32)
        self.Linear3 = nn.Linear(32, 16)
        self.Linear4 = nn.Linear(16, output_dim)
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
        x = self.sigmoid(x)
        return x
model = Model(13,3).to(DEVICE)      
 
 #3. 컴파일, 훈련
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(model, criterion, optimizer, x_train, y_train):
    optimizer.zero_grad()
    hypothesis = model(x_train)
    loss = criterion(hypothesis, y_train)
    loss.backward()
    optimizer.step()
    return loss.item()

epochs = 100
for epoch in range(1, epochs+1):
    loss = train(model, criterion, optimizer, x_train, y_train)
    print('epochs : {}, loss : {}'.format(epoch,loss))
    
#4. 평가, 예측
print("==================")    

def evaluate(model, criterion, x_test, y_test):
    model.eval()
    
    with torch.no_grad():
        y_predict = model(x_test)
        results = criterion(y_predict, y_test)
    return results.item()

loss = evaluate(model, criterion, x_test, y_test)
print(loss)

# y_predict = model(x_test)  
# print(y_predict[:10])    # true/ false 형태로 반환

y_predict = (model(x_test) >= 0.5).float()
print(y_predict[:10])    # 0 또는 1 형태로 반환
y_predict = torch.argmax(y_predict, axis=1) 

score = (y_predict == y_test).float().mean()
print('accuracy : {:.4f}'.format(score))

from sklearn.metrics import accuracy_score
# score = accuracy_score(y_test(), y_predict()) 
score = accuracy_score(y_test.cpu(), y_predict.cpu()) #cpu 명시 필요
print('accuracy score : ',round(score,4))

'''
accuracy : 1.0000
accuracy score :  1.0
'''