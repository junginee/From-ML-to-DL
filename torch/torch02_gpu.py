import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')
print('torch : ',torch.__version__, '사용 DEVICE : ', DEVICE)  

#torch :  1.12.1 사용 DEVICE :  cuda

'''
cuda에 USE_CUDA 설치되어 있으면 gpu, 없으면 cpu로 돌려라
'''

#1. 데이터
# x = np.array([1,2,3])
# y = np.array([1,2,3])

# x = torch.FloatTensor(x).unsqueeze(1).to(DEVICE)
# y = torch.FloatTensor(y).unsqueeze(-1).to(DEVICE)

# print(x,y)
# print(x.shape, y.shape)

# #2. 모델구성
# # model = Sequential()
# model = nn.Linear(1,1).to(DEVICE)

# #3. 컴파일, 훈련
# criterion = nn.MSELoss() 
# optimizer = optim.SGD(model.parameters(), lr=0.01) 

# def train(model,  criterion, optimizer, x, y):
#     # model. train()            # 훈련모드 (생략할 수도 있음)
#     optimizer.zero_grad()   
    
#     hypothesis = model(x)
    
#     loss = criterion(hypothesis, y) 
    
#     loss.backward()
#     optimizer.step()
#     return loss.item()

# epochs = 1100000
# for epoch in range(1, epochs+1) :
#         loss = train(model, criterion, optimizer, x , y)
#         print("epoch : {}, loss: {}".format(epoch, loss))

# #4. 평가, 예측
# # loss = model.evaluate(x, y)
# def evaluate(model, criterion, x, y ):
#         model.eval()        # 평가모드 (반드시 명시 必)
        
#         with torch.no_grad():   
#             x_predict = model(x)
#             results = criterion(x_predict, y)
#         return results.item()

# loss2 = evaluate(model, criterion, x, y)    
# print("최종 loss : ", loss2)

# # y_predict = model.predict([4])    
# results = model(torch.Tensor([[4]])).to(DEVICE)
# print('4의 예측값 : ',results.item())