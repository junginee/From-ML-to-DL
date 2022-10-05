from tensorflow.python.keras.models import Sequential, Model, load_model
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from tensorflow.python.keras.layers import Conv1D, LSTM, Reshape
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils import validation

(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)        
print(x_train.shape, y_train.shape)    
print(x_test.shape, y_test.shape)      


x_train = x_train.reshape(60000, 28, 28, 1) 
x_test = x_test.reshape(10000, 28, 28, 1)                          
                                         

print(np.unique(y_train, return_counts=True))   #10개 
print(x_train.shape, y_train.shape) #(60000, 28, 28, 1) (60000,)


# 2. 모델구성
model  =  Sequential() 
model.add(Conv2D(64, kernel_size=(3,3), padding = 'same', input_shape=(28, 28, 1 ) ))
model.add(MaxPooling2D())                                     #(None, 14, 14, 64) 
model.add(Conv2D(32, (3,3)))                                  #(None, 12, 12, 32) 
model.add(Reshape(target_shape=(32,144))) 
model.add(LSTM(10))                                   
#model.add(Flatten())                                         #(None, 700)
model.add(Dense(100, activation='relu'))                      #(None, 100) 
model.add(Reshape(target_shape=(100,1)))                      #(None, 100, 1)   #순서, 내용은 바뀌지 않음 / 연산량 (X)
model.add(Conv1D(10,kernel_size =3, padding='same'))                                       #(None, 98, 10)
model.add(LSTM(16))                                           #(None, 16) 
model.add(Dense(32, activation="relu"))                       #(None, 32)
model.add(Dense(32, activation="relu"))                       #(None, 32) 
model.add(Dense(10, activation='softmax'))                    #(None, 10)


model.summary()

'''
# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'] )              
# Fit
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',patience=20, mode='auto',
                   verbose=1, restore_best_weights=False)    

model.fit(x_train, y_train, epochs=16, batch_size=32, verbose=1,   
          validation_split=0.3, callbacks=[es])

# 4. 평가, 예측
# Evaluate
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])             # 값이 2개가 나오는데 첫째로 로스가 나오고, 둘째로 accuracy가 나온다.
print('accuracy : ', loss[1])              # ★ accuracy빼고싶을때 loss[0]하면 리스트에서 첫번째만 출력하니까 로스만 찍을 수 있음★


# loss :  0.06993652880191803
# accuracy :  0.9815999865531921
'''
