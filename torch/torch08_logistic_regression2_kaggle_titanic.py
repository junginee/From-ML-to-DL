from sklearn.datasets import load_digits
import pandas as pd
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn 
import torch.optim as optim
from sklearn.model_selection import train_test_split


USE_CUDA = torch.cuda.is_available()
DEVICE  = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('torch:', torch.__version__,'사용DEVICE :',DEVICE)

path = './_data/kaggle_titanic/'
train_set = pd.read_csv(path +'train.csv')
test_set = pd.read_csv(path + 'test.csv',index_col=0) 

print(train_set.Pclass.value_counts()) 

Pclass1 = train_set["Survived"][train_set["Pclass"] == 1].value_counts(normalize = True)[1]*100
Pclass2 = train_set["Survived"][train_set["Pclass"] == 2].value_counts(normalize = True)[1]*100
Pclass3 = train_set["Survived"][train_set["Pclass"] == 3].value_counts(normalize = True)[1]*100
print(f"Percentage of Pclass 1 who survived: {Pclass1}")
print(f"Percentage of Pclass 2 who survived: {Pclass2}")
print(f"Percentage of Pclass 3 who survived: {Pclass3}")

female = train_set["Survived"][train_set["Sex"] == 'female'].value_counts(normalize = True)[1]*100
male = train_set["Survived"][train_set["Sex"] == 'male'].value_counts(normalize = True)[1]*100
print(f"Percentage of females who survived: {female}")
print(f"Percentage of males who survived: {male}")

sns.barplot(x="SibSp", y="Survived", data=train_set)

train_set = train_set.fillna({"Embarked": "S"})
train_set.Age = train_set.Age.fillna(value=train_set.Age.mean())

train_set = train_set.drop(['Name'], axis = 1)
test_set = test_set.drop(['Name'], axis = 1)

train_set = train_set.drop(['Ticket'], axis = 1)
test_set = test_set.drop(['Ticket'], axis = 1)

train_set = train_set.drop(['Cabin'], axis = 1)
test_set = test_set.drop(['Cabin'], axis = 1)

train_set = pd.get_dummies(train_set,drop_first=True)
test_set = pd.get_dummies(test_set,drop_first=True)

test_set.Age = test_set.Age.fillna(value=test_set.Age.mean())
test_set.Fare = test_set.Fare.fillna(value=test_set.Fare.mode())

print(train_set, test_set, train_set.shape, test_set.shape)

x = train_set.drop(['Survived', 'PassengerId'], axis=1)  
print(x)
print(x.columns)
print(x.shape) # (891, 8)

y = train_set['Survived'] 
print(y)
print(y.shape) # (891,)

x = torch.FloatTensor(x.values)
y = torch.FloatTensor(y.values)

x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.7,
                                                    random_state=66
                                                    )

print(x_train.size(), x_test.size(), y_train.size(), y_test.size()) 

x_train = torch.FloatTensor(x_train)
x_test = torch.FloatTensor(x_test)
y_train = torch.FloatTensor(y_train).unsqueeze(-1).to(DEVICE)
y_test = torch.FloatTensor(y_test).unsqueeze(-1).to(DEVICE)

print(x_train.size(), x_test.size(), y_train.size(), y_test.size()) 

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

print('################scaler 후##################')

print('x_trian:',x_train)  
print('x_test:',x_test) 

x_train = torch.FloatTensor(x_train).to(DEVICE)
x_test = torch.FloatTensor(x_test).to(DEVICE)

print(x_train.size()) #torch.Size([623, 8])
print(x_train.shape)  #torch.Size([623, 8])


#2. 모델구성
model  = nn.Sequential(
    nn.Linear(8 ,64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
    nn.Sigmoid(),
).to(DEVICE)

#3. 컴파일, 훈련
criterion = nn.BCELoss().to(DEVICE) 

optimizer  = optim.Adam(model.parameters(), lr=0.01) 

def train(model, criterion , optimizer , x_train, y_train):
    model.train() 
    optimizer.zero_grad()    
    hypothesis = model(x_train)
    loss = criterion(hypothesis, y_train) 
    loss.backward() 
    optimizer.step()
    return loss.item()  

EPOCHS = 100
for epoch in range(1,EPOCHS + 1):   
    loss = train(model, criterion , optimizer , x_train, y_train)
    print('epoch {}, loss: {:.8f}'.format(epoch, loss)) 


#4. 평가, 예측
print('======================평가, 예측======================')

def evaluate(model, criterion, x_test, y_test):  
    model.eval()

    with torch.no_grad():
        y_predict = model(x_test)
        loss = criterion(y_predict, y_test)
    return loss.item()

loss = evaluate(model, criterion, x_test, y_test) 
print('최종 loss : ',loss) 

y_predict = (model(x_test) >=0.5).float() 
print(y_predict[:10])

score = (y_predict == y_test).float().mean() 
print('accuracy:,{:.4f}'.format(score))

from sklearn.metrics import accuracy_score
score = accuracy_score(y_test.cpu().numpy(), y_predict.cpu().numpy())  
print('accuracy_score:',(score))

'''
accuracy:,0.8433
accuracy_score: 0.8432835820895522
'''
