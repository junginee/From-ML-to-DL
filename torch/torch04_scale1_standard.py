
from inspect import Parameter
from pickletools import optimize
from unittest import result
import numpy as np 
import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')  
print('torch : ', torch.__version__,'사용DEVICE : ', DEVICE)

# 1.데이터
x = np.array([1,2,3])   
y = np.array([1,2,3])   
x_test = np.array([4])

x = torch.FloatTensor(x).unsqueeze(1).to(DEVICE)  
y = torch.FloatTensor(y).unsqueeze(1).to(DEVICE)
x_test = torch.FloatTensor(x_test).unsqueeze(1).to(DEVICE)

x = (x - torch.mean(x) / torch.std(x)) # standard scaler
x_test =(x_test -torch.mean(x) / torch.std(x))

print(x,y)
print(x.shape,y.shape)

# 2.모델
# model = Sequential()
model = nn.Linear(1, 1).to(DEVICE) # (인풋 x값 , 아웃풋 y값)

# 3.컴파일,훈련 
# model.compile(loss='mse',optimizer='SGD')
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01)
# optim.Adam(model.parameters(), lr = 0.01)

def train(model, criterion, optimizer, x, y):
    # model.train()         
    optimizer.zero_grad()    
    
    hypothesis = model(x)    
    
    # loss = criterion(hypothesis, y)
    # loss = nn.MSELoss()(hypothesis, y)  
    loss = F.mse_loss(hypothesis, y)
    
    loss.backward()                                      
    optimizer.step()                                         
    return loss.item()

epochs = 2000
for epoch in range(1, epochs+1):
    loss = train(model, criterion, optimizer, x, y)
    print('epochs : {}, loss: {}'.format(epoch,loss))    
    
# 4.평가, 예측
# loss = model.evaluate(x,y)

def evaluate(model, criterion, x, y):
    model.eval()                # 평가모드 
    
    with torch.no_grad():    
        y_predict = model(x)
        results = criterion(y_predict,y)
    return results.item()

loss2 = evaluate(model, criterion, x, y)
print('최종 loss : ', loss2)

# y_predict = model.predict([4])

results = model(torch.Tensor(x_test).to(DEVICE)) #2차원으로 넣어줘야함.
print('predict의 예측값 : ', results.item())
