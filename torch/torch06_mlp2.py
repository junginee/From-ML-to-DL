import numpy as np 
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


#1.데이터
x = np.array([[1,2,3,4,5,6,7,8,9,10],
              [1, 1, 1, 1, 2, 1.3, 1.4, 1.5, 1.6, 1.4],
              [9,8,7,6,5,4,3,2,1,0]]) 
y = np.array([11,12,13,14,15,16,17,18,19,20]) 
x_test = np.array([10, 1.4, 0])

print(x.shape,y.shape,x_test.shape)

x = torch.FloatTensor(x).to(DEVICE)
y = torch.FloatTensor(y).unsqueeze(-1).to(DEVICE)
x_test = torch.FloatTensor(x_test).unsqueeze(-1).to(DEVICE)

x = x.T
x_test = x_test.T

print(x.shape,y.shape,x_test.shape)

# 스케일링 #
x_test = (x_test - torch.mean(x)) / torch.std(x) 
x = (x - torch.mean(x)) / torch.std(x)

#2.모델구성
model = nn.Sequential(
    nn.Linear(3,10),
    nn.ReLU(),
    nn.Linear(10,10),
    nn.ReLU(),
    nn.Linear(10,10),
    nn.ReLU(),
    nn.Linear(10,5),
    nn.Linear(5,1)
).to(DEVICE)

#3.컴파일,훈련
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.01)

def train(model,criterion,optimizer,x,y):
    optimizer.zero_grad()
    
    hypothesis = model(x)
    
    loss = criterion(hypothesis,y)
    
    loss.backward()
    optimizer.step()
    return loss.item()

epochs = 2000
for epoch in range(1, epochs+1):
    loss = train(model, criterion, optimizer, x, y)
    print('epochs : {}, loss : {}'.format(epoch,loss))    

#4.평가,예측
def evaluate(model, criterion, x, y):
    model.eval()
    
    with torch.no_grad():
        y_predict = model(x)
        results = criterion(y_predict,y)
    return results.item()

loss2 = evaluate(model, criterion, x, y)
print('최종 loss : ', round(loss2,7))

results = model(torch.Tensor(x_test).to(DEVICE))
print('x_test의 결과값 : ', round(results.item(),4))

'''
최종 loss :  0.0004156
x_test의 결과값 :  20.0407
'''