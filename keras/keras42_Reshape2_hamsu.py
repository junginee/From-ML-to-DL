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
