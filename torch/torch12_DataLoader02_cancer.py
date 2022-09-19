

from sklearn.datasets import load_breast_cancer
import torch
import torch.nn as nn
import torch.optim as optim 

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')  
print('torch : ', torch.__version__,'사용DEVICE : ', DEVICE)

#. 1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets['target']

x = torch.FloatTensor(x)
y = torch.FloatTensor(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                    train_size=0.7, shuffle = True, random_state=123, stratify=y)

x_train = torch.FloatTensor(x_train)
y_train = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
x_test = torch.FloatTensor(x_test)
y_test = torch.FloatTensor(y_test).unsqueeze(-1).to(DEVICE)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)

print(x_train.size())  #(=shape) torch.Size([398, 30])
print(x_train.shape) 

############################
from torch.utils.data import TensorDataset, DataLoader
train_set = TensorDataset(x_train, y_train)     #x, y를 합친다!
test_set = TensorDataset(x_test, y_test)         #x, y를 합친다!

print(train_set) #<torch.utils.data.dataset.TensorDataset object at 0x000002F16E3E3F70> 
print("============train_set[0]==============")
print(train_set[0])         # x, y train_set
print("============train_set[0][0]==============")
print(train_set[0][0])      # x
print("============train_set[0][1]==============")
print(train_set[0][1])      # y
print("============len(train_set)==============")
print(len(train_set))   #398


##########x, y 배치 합체#########
train_loader = DataLoader(train_set, batch_size=40, shuffle=True)
test_loader = DataLoader(test_set, batch_size=40, shuffle=True)
#############################

'''
class = 모델정의
forward(순전파)에서 모델 실행
Model은 nn.module 함수, 변수를 상속받겠다.
즉 모듈을 상속받음 (상위: 모듈, 하위: 모델)
'''

class Model(nn.Module): 
        def __init__(self, input_dim, output_dim):
            # super().__init__()                     # 방법1
            super(Model, self).__init__()       # 방법2
            self.linear1 = nn.Linear(input_dim, 64)
            self.linear2 = nn.Linear(64, 32 )
            self.linear3 = nn.Linear(32, 16 )
            self.linear4 = nn.Linear(16, output_dim )
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
                        
        def forward(self, input_size):    
            x = self.linear1(input_size)
            x = self.relu(x)
            x = self.linear2(x)
            x = self.relu(x)
            x = self.linear3(x)
            x = self.relu(x)
            x = self.linear4(x)
            x = self.sigmoid(x)
            return x
model = Model(30,1).to(DEVICE)       

#3. 컴파일, 훈련
criterion = nn.BCELoss()

optimizer = optim.Adam(model.parameters(), lr=0.01)

################## 배치 작업 #################
def train(model, criterion, optimizer, loader):
    total_loss = 0
    for x_batch, y_batch in loader:      
        optimizer.zero_grad()
        hypothesis = model(x_batch)
        loss = criterion(hypothesis, y_batch)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)

''''
len(loader) 필수 기재 할 필요 x
return 값 : total loss 를 loader 갯수로 나누어 줌(398/10 = 3.98 = 4)
'''
###########################################

epochs = 100
for epoch in range(1, epochs+1):
    loss = train(model, criterion, optimizer, train_loader)
    if epoch % 10 == 0 :
        print('epochs : {}, loss : {:.8f}'.format(epoch,loss))
    
#4. 평가, 예측
print("==== ==============")    

def evaluate(model, criterion, loader):
    model.eval()
    total_loss = 0
    
    for x_batch, y_batch in loader:
        with torch.no_grad():
            y_predict = model(x_batch)
            results = criterion(y_predict, y_batch)
        return results.item()

loss = evaluate(model, criterion, test_loader)
print(loss)

y_predict = (model(x_test) >= 0.5).float()
print(y_predict[:10])    

'''
predict 시 loader 데이터와 x_test 데이터 shape 같으므로 변경 x
'''

score = (y_predict == y_test).float().mean()
print('accuracy : {:.4f}'.format(score))

from sklearn.metrics import accuracy_score
score = accuracy_score(y_test.cpu(), y_predict.cpu()) 
print('accuracy score : ',round(score,4))

'''
accuracy : 0.9766
accuracy score :  0.9766
'''