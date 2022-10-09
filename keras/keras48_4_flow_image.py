from tensorflow.keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D,Flatten
from sklearn.model_selection import train_test_split
import numpy as np


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


train_datagen = ImageDataGenerator(
    rescale = 1./255,
    horizontal_flip = True,
    width_shift_range = 0.5,
    height_shift_range = 0.5,
    zoom_range = 0.3,
    fill_mode = 'nearest')


test_datagen = ImageDataGenerator(
    rescale = 1./255
)

augment_size = 10
randidx = np.random.randint(x_train.shape[0], size=augment_size)

# print(x_train.shape[0])                 
# print(randidx)                         
# print(np.min(randidx), np.max(randidx)) 

x_train1 = x_train[randidx].copy()
x_augmented = x_train[randidx].copy()
y_augmented = y_train[randidx].copy()

print(x_augmented.shape)                # (10, 28, 28)  
print(x_augmented.shape[0],x_augmented.shape[1],x_augmented.shape[2]) # 각 10 28 28

# (?, ?, ?) 에서 (?, ?, ?, ?)로.
x_augmented = x_augmented.reshape(x_augmented.shape[0],
                                  x_augmented.shape[1],
                                  x_augmented.shape[2], 1)
x_train1 = x_train1.reshape(x_train1.shape[0], 28, 28, 1)
print(x_train1.shape) #(10, 28, 28, 1)


x_augmented = train_datagen.flow(x_augmented, y_augmented,
                                 batch_size = augment_size, shuffle=False,
                                 ).next()[0] # x 데이터 증폭


x_train = np.concatenate((x_train1, x_augmented))    
print(x_train.shape) #(20, 28, 28, 1)


# 과제
# x_augment 10개와 x_train1 10개를 비교하는 이미지를 출력할 것(즉, 변환 전 후 비교 )(위 10개 아래 10개로)(순서는 randidx순서)
# subplot(2,10,? ) 사용      # nrows=2, ncols=10, index=?

# print(x_augmented)

#print(type(x_augmented)) # <class 'numpy.ndarray'>

x_data = test_datagen.flow(
    x_train.reshape(20,28,28,1), # x
    np.zeros(20),                # y는 x[0] shape에 맞춰서 20개로 설정
    batch_size=20,
    shuffle=False,
).next()[0]

import matplotlib.pyplot as plt
plt.figure(figsize=(5,5))
plt.title('변환 전 후 비교')
for i in range(20):
    plt.subplot(2, 10, i+1)
    plt.axis('off')
    plt.imshow(x_train[i], cmap='winter_r')
  
plt.show()   
