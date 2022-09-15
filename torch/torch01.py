import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

#1. 데이터
x = np.array([1,2,3]) 
y = np.array([1,2,3])
x_test = np.array([[[4]]]) #(1, 1)
print(x_test.shape)

#토치에서 1차원으로 데이터 사용하면 오류날 수도 있음, 2차원으로 reshape
#정수 tensor 형태
x = torch.FloatTensor(x).unsqueeze(1)   #(3,) => (3,1) (방법1)
y = torch.FloatTensor(y).unsqueeze(-1)  #(3,) => (3,1) (방법2)

print(x,y)
print(x.shape, y.shape)

#2. 모델구성
# model = Sequential()
model = nn.Linear(1,1) #input x, output y / 단층레이어, 선형회귀

#3. 컴파일, 훈련
# model.compile(loss='mes', optimizer='SGD')
criterion = nn.MSELoss() #표준
optimizer = optim.SGD(model.parameters(), lr=0.01) 
옵티마이저는 모든 파라미터에서 연산되기 때문에 model.parameters 명시
optim.Adam(model.parameters(), lr=0.01)

def train(model,  criterion, optimizer, x, y):
    # model. train()             # 훈련모드 (생략할 수도 있음)
    optimizer.zero_grad()   #손실함수 기울기 초기화
    
    hypothesis = model(x)
    
    loss = criterion(hypothesis, y) #mse(hypothesis - y) = loss 
    
    loss.backward()
    optimizer.step()
    return loss.item()

epochs = 700
for epoch in range(1, epochs+1) :
        loss = train(model, criterion, optimizer, x , y)
        print("epoch : {}, loss: {}".format(epoch, loss))
'''
파이토치 반복 문법
    1.optimizer.zero_grad()   #손실함수 기울기 초기화
    2.loss.backward() 역전파
    3.optimizer.step 역전파를 하면서 웨이트 갱신
    1 epoch = 1-> 2-> 3
'''    

#4. 평가, 예측
# loss = model.evaluate(x, y)

# 평가는 만들어진 가중치로만 평가하기 때문에 가중치 갱신할 필요 없음 => optimizer 안들어가도 됨
def evaluate(model, criterion, x, y ):
        model.eval()        #평가모드 (반드시 명시 必)
        
        with torch.no_grad():   
            x_predict = model(x)
            results = criterion(x_predict, y)
        return results.item()

loss2 = evaluate(model, criterion, x, y)    
print("최종 loss : ", loss2)

#y_predict = model.predict([4])    
results = model(torch.Tensor([[4]])) #shape 맞춰주기 (1,1)
print('4의 예측값 : ',results.item())