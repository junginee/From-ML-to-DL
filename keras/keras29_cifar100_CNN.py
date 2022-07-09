from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D  # 1D는 선만 그어. 2D부터 이미지
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils import validation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

scaler = StandardScaler()

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

y_train = to_categorical(y_train)     

y_test = to_categorical(y_test)       


n = x_train.shape[0]# 이미지갯수 50000
x_train_reshape = x_train.reshape(n,-1) #----> (50000,32,32,3) --> (50000, 32*32*3 ) 0~255
x_train_transe = scaler.fit_transform(x_train_reshape) #0~255 -> 0~1

x_train = x_train_transe.reshape(x_train.shape) #--->(50000,32,32,3) 0~1

m = x_test.shape[0]
x_test = scaler.transform(x_test.reshape(m,-1)).reshape(x_test.shape)
# print(x_train.shape, y_train.shape)     # (50000, 32, 32, 3) (50000, 10)
# print(x_test.shape, y_test.shape)       # (10000, 32, 32, 3) (10000, 10)



# print(x_train[0])
# print('y_train[0]번째 값 : ', y_train[0])

# import matplotlib.pyplot as plt
# plt.imshow(x_train[0], 'gray')
# plt.show()
# print(np.unique(y_train, return_counts=True))


# 2. 모델구성
#hint conv-layer는 3~4개
model  =  Sequential() 
model.add(Conv2D(10, kernel_size=(3,3), strides=1, padding='same', input_shape=(32, 32, 3) )) 
# model.add(Conv2D(7, kernel_size=(3,3), input_shape=(28, 28, 1 ) ))
model.add(MaxPooling2D()) 
model.add(Conv2D(5, (3,3), activation="relu") )
model.add(Dropout(0.3))
model.add(Conv2D(7, (3,3), activation="relu") )
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(32, activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(10, activation='softmax'))




# 3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'] ) 

# Fit
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',patience=20, mode='auto',
                   verbose=1, restore_best_weights=False)    

model.fit(x_train, y_train, epochs=20, batch_size=32, verbose=1,    
          validation_split=0.3, callbacks=[es])


# 4. 평가, 예측
# Evaluate
loss = model.evaluate(x_test, y_test)
print('loss : ', loss[0])             # 값이 2개가 나오는데 첫째로 loss가 나오고, 둘째로 accuracy가 나온다.
print('accuracy : ', loss[1])              # ★ accuracy빼고싶을때 loss[0]하면 리스트에서 첫번째만 출력하니까 로스만 찍을 수 있음★

# loss :  1.372598648071289   accuracy :  0.5202000141143799
