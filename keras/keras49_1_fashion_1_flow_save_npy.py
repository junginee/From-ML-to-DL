

from tensorflow.keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import  Conv2D, MaxPool2D, Flatten, Dense, Dropout
import numpy as np
import pandas as pd

#1. 데이터


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
randidx = np.random.randint(x_train.shape[0], size=augument_size) 

x_augumented = x_train[randidx].copy()
y_augumented = y_train[randidx].copy() 

x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
x_augumented = x_augumented.reshape(x_augumented.shape[0], 
                                    x_augumented.shape[1], 
                                    x_augumented.shape[2], 1)

import time
start_time=time.time()
print('시작')
x_augumented = train_datagen.flow(x_augumented, y_augumented,
                                  batch_size=augument_size,
                                  save_to_dir='D:\study_data\_temp'
                                  shuffle=False).next()[0] 

end_time = time.time()-start_time    
print(augument_size, '개 증폭에 걸린시간 :', round(end_time,3), "초")
test_datagen = ImageDataGenerator(
    rescale = 1./255
    
)

xy_train = train_datagen.flow_from_directory(
    'd:/study_data/_data/image/brain/train/',
    target_size=(150,150),
    batch_size=500,
    class_mode = 'binary',
    color_mode = 'grayscale',
    shuffle=True,
)  #Found 160 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    'd:/study_data/_data/image/brain/test/',
    target_size=(150,150), #이미지 사이즈 동일하지 않기 때문에 일괄적으로 150,150으로 고정
    batch_size=500,
    class_mode = 'binary',
    color_mode = 'grayscale',
    shuffle=True,
)  #Found 120 images belonging to 2 classesv.


# print(xy_train[0]) #마지막 배치  / [0]-[1]-[2] ...[31] 까지
# print(xy_train[2][0]) #뒤에 [0]로 하면 x값
# print(xy_train[2][0]) #뒤에 [1]로 하면 y값
# print(xy_train[31][2]) #error

print(xy_train[0][0].shape) #(160, 150, 150, 1)
print(xy_train[0][1].shape) #(160,)
print(xy_test[0][0].shape)  #(120, 150, 150, 1)
print(xy_test[0][1].shape)  #(120,)

np.save('d:/study_data/_save/_npy/keras46_5_train_x.npy', arr=xy_train[0][0])
np.save('d:/study_data/_save/_npy/keras46_5_train_y.npy', arr=xy_train[0][1])
np.save('d:/study_data/_save/_npy/keras46_5_test_x.npy', arr=xy_test[0][0])
np.save('d:/study_data/_save/_npy/keras46_5_test_y.npy', arr=xy_test[0][1])