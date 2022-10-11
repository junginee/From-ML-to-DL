from warnings import filters
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Activation, Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, accuracy_score
from keras.preprocessing.image import ImageDataGenerator

#1. 데이터
train_datagen = ImageDataGenerator(
    horizontal_flip=True,
    # vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    # shear_range=0.7,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator()

xy_train = train_datagen.flow_from_directory(
    'd:/study_data/_data/image/cat_dog/training_set/',
    target_size=(100,100),
    batch_size=8005,
    class_mode='categorical',
    shuffle=False
) #Found 8005 images belonging to 2 classes.


xy_test = test_datagen.flow_from_directory(
    'd:/study_data/_data/image/cat_dog/test_set/',
    target_size=(100,100),
    batch_size=2023,
    class_mode='categorical',
    shuffle=False
) #Found 2023 images belonging to 2 classes.

print(xy_train) 
#<keras.preprocessing.image.DirectoryIterator object at 0x000002C22310F9D0>

# print(xy_train[0][0]) # 마지막 배치
print(xy_train[0][0].shape,xy_train[0][1].shape)
# print(xy_train[0][1])
print(xy_test[0][0].shape,xy_test[0][1].shape)

x_train = xy_train[0][0]
y_train = xy_train[0][1]
x_test = xy_test[0][0]
y_test = xy_test[0][1]


print(x_train.shape,x_test.shape) #(8005, 100, 100, 3) (2023, 100, 100, 3)
print(y_train.shape,y_test.shape) #(8005, 2) (2023, 2)



#################################### 스케일링 ######################################
x_train1 = x_train.reshape((x_train.shape[0]), (x_train.shape[1])*(x_train.shape[2])*3)
x_test1 = x_test.reshape((x_test.shape[0]), (x_test.shape[1])*(x_test.shape[2])*3)

scaler = MinMaxScaler()
x_train1 = scaler.fit_transform(x_train1)
x_test1 = scaler.transform(x_test1)

x_train = x_train1.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 3)
x_test = x_test1.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 3)

###################################################################################

augument_size = 5 # 증폭
randindx = np.random.randint(x_train.shape[0], size = augument_size)
print(randindx,randindx.shape) # (40000,)
print(np.max(randindx), np.min(randindx)) # 59997 2
print(type(randindx)) # <class 'numpy.ndarray'>

x_augumented = x_train[randindx].copy()
print(x_augumented,x_augumented.shape) # (40000, 28, 28, 1)
y_augumented = y_train[randindx].copy()
print(y_augumented,y_augumented.shape) # (40000,)

# x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
# x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
# x_augumented = x_augumented.reshape(x_augumented.shape[0], 
#                                     x_augumented.shape[1], x_augumented.shape[2], 1)

x_augumented = train_datagen.flow(x_augumented, y_augumented,
                                  batch_size=augument_size,
                                  shuffle=False).next()[0]
print(x_augumented[0][1])

x_train = np.concatenate((x_train, x_augumented))
y_train = np.concatenate((y_train, y_augumented))



xy_train = test_datagen.flow(x_train, y_train,
                                  batch_size=100000,
                                  shuffle=False)

print(xy_train[0][0])
print(xy_train[0][0].shape)

print(xy_train[0][0].shape) #(200, 150, 150, 1)
print(xy_train[0][1].shape) #(200,)

np.save('d:/study_data/_save/_npy/keras49_6_train_x.npy', arr=xy_train[0][0])
np.save('d:/study_data/_save/_npy/keras49_6_train_y.npy', arr=xy_train[0][1])
np.save('d:/study_data/_save/_npy/keras49_6_test_x.npy', arr=x_test)
np.save('d:/study_data/_save/_npy/keras49_6_test_y.npy', arr=y_test)
