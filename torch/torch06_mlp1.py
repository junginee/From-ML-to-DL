# 실습


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
x = np.array([[1,2,3,4,5,6,7,8,9,10],
                       [1,1.1,1.2,1.3,1.4,1.5,1.6,1.5,1.4,1.3]] )    
y = np.array([11,12,13,14,15,16,17,18,19,20])   
x_test = np.array([10, 1.3])

print(x.shape,y.shape) #(2, 10), (10,)

x = x.T
x = torch.FloatTensor(x).to(DEVICE)
y = torch.FloatTensor(y).unsqueeze(1).to(DEVICE)
x_test = torch.FloatTensor(x_test).unsqueeze(1).to(DEVICE)    

print(x.shape) #torch.Size([10, 2])  >> input : 2
print(y.shape) #torch.Size([10, 1]) >> output : 1 
print(x_test.shape) #torch.Size([2, 1])

x = (x - torch.mean(x) / torch.std(x)) 
x_test= (x_test - torch.mean(x) / torch.std(x))


# 2.모델
# model = Sequential()

'''
model = nn.Linear(1, 1).to(DEVICE) # (인풋 x값 , 아웃풋 y값)
model = nn.Linear(5, 3).to(DEVICE)
model = nn.Linear(3, 4).to(DEVICE)
model = nn.Linear(4, 2).to(DEVICE)
model = nn.Linear(2, 1).to(DEVICE)
'''

model = nn.Sequential(
  nn.Linear(2,4), 
  nn.Linear(4,5), 
  nn.Linear(5,3), 
  nn.Linear(3,2), 
  nn.Linear(2,1) 
  ) .to(DEVICE)

# 3.컴파일,훈련 
# model.compile(loss='mse',optimizer='SGD')
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01)
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

epochs = 500
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
# predict = (x_test - torch.mean(x) / torch.std(x)) 

results = model(torch.Tensor([[np.array([10, 1.3])]]).to(DEVICE)) 
print('predict의 예측값 : ', round(results.item(),4)) 

# 최종 loss :  0.004872085992246866
# predict의 예측값 :  20.6741