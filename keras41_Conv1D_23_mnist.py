import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D,LSTM,Conv1D
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler, RobustScaler
from tensorflow.python.keras.callbacks import EarlyStopping

#1.데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

x_train = x_train.reshape(60000,28*28)
x_test = x_test.reshape(10000,28*28)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(10000,28,28,1)
print(x_train.shape)
import numpy as np
print(np.unique(y_train,return_counts=True))

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print(x_train.shape,x_test.shape)
x_train = x_train.reshape(60000,28*28,1)
x_test = x_test.reshape(10000,28*28,1)

#2.모델구성
model = Sequential()
model.add(Conv1D(8,2,activation='relu', input_shape=(28*28,1))) 
model.add(Flatten())
model.add(Dense(6,activation= 'relu'))
model.add(Dense(4,activation= 'relu'))
model.add(Dense(2,activation= 'relu'))
model.add(Dense(10,activation='softmax'))




# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'] )              
# Fit
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',patience=20, mode='auto',
                   verbose=1, restore_best_weights=False)    

model.fit(x_train, y_train, epochs=100, batch_size=50, verbose=1,   
          validation_split=0.3, callbacks=[es])

# 4. 평가, 예측
# Evaluate
loss = model.evaluate(x_test, y_test)
print('loss : ', round(loss[0],4))             # 값이 2개가 나오는데 첫째로 로스가 나오고, 둘째로 accuracy가 나온다.
print('accuracy : ', round(loss[1],4))              # ★ accuracy빼고싶을때 loss[0]하면 리스트에서 첫번째만 출력하니까 로스만 찍을 수 있음★
print('MNIST')

# [LSTM]
# loss :  0.7812
# accuracy :  0.727

# [CONV1]
# loss :  0.5532
# accuracy :  0.8766