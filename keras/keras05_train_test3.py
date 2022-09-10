import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10]) 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, train_size=0.7, shuffle=True, random_state=66)

print(x_train) #[2 7 6 3 4 8 5]
print(x_test) #[ 1  9 10]
print(y_train) #[2 7 6 3 4 8 5]
print(x_test) #[ 1  9 10]

#2. 모델구성 
model = Sequential()
model.add(Dense(5,input_dim=1)) 
model.add(Dense(5)) 
model.add(Dense(5))
model.add(Dense(7))
model.add(Dense(9))  
model.add(Dense(1))  

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit( x_train, y_train, epochs=300, batch_size=1)

#4. 평가, 예측
loss = model.evaluate(x_test,y_test,)
print('loss :',loss)
result = model.predict([11]) 
print('11의 예측값:', result)
