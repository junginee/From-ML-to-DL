import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

# 1. 데이터
x_train = np.array([1,2,3,4,5,6,7]) # (7,)
x_test = np.array([8,9,10])         # (3,)
y_train = np.array([1,2,3,4,5,6,7]) # (7,)
y_test = np.array([8,9,10])         # (3,)

x_predict = np.array([11,12,13])

x_train = torch.Tensor(x_train).unsqueeze(-1).to(DEVICE)
x_test = torch.Tensor(x_test).unsqueeze(-1).to(DEVICE)
y_train = torch.Tensor(y_train).unsqueeze(-1).to(DEVICE)
y_test = torch.Tensor(y_test).unsqueeze(-1).to(DEVICE)
x_predict = torch.Tensor(x_predict).unsqueeze(-1).to(DEVICE)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
print(x_predict.shape)

#### 스케일링 ####
x_predict = (x_predict - torch.mean(x_train)) / torch.std(x_train)
x_test = (x_test - torch.mean(x_train)) / torch.std(x_train)
x_train = (x_train - torch.mean(x_train)) / torch.std(x_train)

# 2. 모델
model = nn.Sequential(
    nn.Linear(1, 16),
    nn.Linear(16, 32),
    nn.ReLU(),
    nn.Linear(32, 64),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.Linear(16, 1),
).to(DEVICE)

# 3. 컴파일, 훈련
optimizer = optim.SGD(model.parameters(), lr=0.001)

def train(model, optimizer, x, y):
    model.train()
    optimizer.zero_grad()
    
    hypothesis = model(x)
    loss = nn.MSELoss()(hypothesis, y)
    
    loss.backward()
    optimizer.step()
    
    return loss.item()

epochs = 1000
for epoch in range(1, epochs+1):
    loss = train(model, optimizer, x_train, y_train)
    print(f'epoch: {epoch}, loss: {loss}')
    
# 4. 평가, 예측
def evaluate(model, x, y):
    model.eval()
    
    with torch.no_grad():
        pred_y = model(x)
        result = nn.MSELoss()(pred_y, y)
        
    return result.item()

loss = evaluate(model, x_test, y_test)
print(f'최종 loss: {loss}')

result = model(x_predict).cpu().detach().numpy()
print('예측값: ' ,'\n ',result)

'''
최종 loss: 0.0342232808470726
예측값:
 [[10.667531]
 [11.588009]
 [12.508491]]
'''