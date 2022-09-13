import numpy as np
from sympy import Max
from keras.datasets import mnist
from tensorflow.python.keras import activations

def autoencoder(hidden_layer_size):
    model = Sequential()
    model.add(Conv2D(hidden_layer_size,kernel_size=(2,2),padding='same',activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(hidden_layer_size*2,(2,2),padding='same',activation='relu'))
    model.add(MaxPooling2D())
    model.add(Conv2D(hidden_layer_size*4,(2,2),padding='same',activation='relu'))
    model.add(UpSampling2D())
    model.add(Conv2D(hidden_layer_size*2,(2,2),padding='same',activation='relu'))
    model.add(UpSampling2D())
    model.add(Conv2D(hidden_layer_size,(2,2),padding='same',activation='relu'))
    model.add(Conv2D(1,(2,2),padding='same',activation='sigmoid'))
    return model

#1. 데이터
(x_train, _) , (x_test, _)  = mnist.load_data()

x_train = x_train.reshape(60000,28,28,1).astype('float')/255   # 또는 /255. 으로 나눠도 된다.
x_test = x_test.reshape(10000,28,28,1).astype('float')/255   

#2. 모델
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, UpSampling2D

model = autoencoder(hidden_layer_size=32)

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(x_train,x_train,epochs=10)

#4. 평가, 예측
output = model.predict(x_test)

# EDA
from matplotlib import pyplot as plt
import random

fig, ((ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10)) = \
    plt.subplots(2,5,figsize = (20,7))                          

# 이미지 5개를 무작위로 고른다.
random_images = random.sample(range(output.shape[0]), 5)

# 원본(입력) 이미지를 맨 위에 그린다.
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow(x_test[random_images[i]].reshape(28,28), cmap = 'gray')
    if i == 0:
        ax.set_ylabel("INPUT", size = 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
 # 오토인코더가 출력한 이미지를 아래에 그린다.   
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow(output[random_images[i]].reshape(28,28), cmap = 'gray')
    if i == 0:
        ax.set_ylabel("OUTPUT", size = 20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()