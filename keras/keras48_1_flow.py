from tensorflow.keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

(x_train, y_train),(x_test, y_test) = fashion_mnist.load_data()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    shear_range=0.7,
    fill_mode='nearest'
)

augument_size = 100

print(x_train[0].shape) #(28, 28)
print(x_train[0].reshape(28*28).shape) #(784,)
print(np.tile(x_train[0].reshape(28*28), augument_size).reshape(-1,28,28,1).shape) 
#np.tile=배열 복사하기, 붙여넣기 함수 (100, 28, 28, 1)

print(np.zeros(augument_size))
print(np.zeros(augument_size).shape) #(100,)

x_data = train_datagen.flow(
    np.tile(x_train[0].reshape(28*28), augument_size).reshape(-1,28,28,1),
    np.zeros(augument_size),
    batch_size=augument_size,
    shuffle=True).next()  # batch_size만큼의 랜덤하게 변형된 학습 데이터를 만든다.

 
print(x_data) #<keras.preprocessing.image.NumpyArrayIterator object at 0x000002A794291D90>


######################## .next() 사용 ####################################
print(x_data[0]) #batch만큼 자른 x, y데이터 출력         
print(x_data[0].shape) #(100, 28, 28, 1)    .next() 사용해 x 출력
print(x_data[1].shape) #(100,)              .next() 사용해 y 출력


######################## .next() 미사용 ###################################
print(x_data[0])          #batch만큼 자른 x, y데이터 출력   
print(x_data[0][0].shape) #(100, 28, 28, 1)   .next() 미사용 x 출력
print(x_data[0][1].shape) #(100,)             .next() 미사용 y 출력

import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
for i in range(100):
  plt.subplot(10,10,i+1)  
  plt.axis('off')
  plt.imshow(x_data[0][0][i], cmap='winter_r')     
plt.show()  
