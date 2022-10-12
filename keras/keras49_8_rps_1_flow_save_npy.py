from warnings import filters
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Activation, Dense, Conv2D, Flatten, MaxPooling2D, Input, Dropout
from keras.datasets import mnist, fashion_mnist, cifar10, cifar100
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical # https://wikidocs.net/22647 케라스 원핫인코딩
from sklearn.preprocessing import OneHotEncoder  # https://psystat.tistory.com/136 싸이킷런 원핫인코딩
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler
from keras.layers import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


###########################폴더 생성시 현재 파일명으로 자동생성###########################################
import inspect, os
a = inspect.getfile(inspect.currentframe()) #현재 파일이 위치한 경로 + 현재 파일 명
print(a)
print(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))) #현재 파일이 위치한 경로
print(a.split("\\")[-1]) #현재 파일 명
current_name = a.split("\\")[-1]
##########################밑에 filepath경로에 추가로  + current_name + '/' 삽입해야 돌아감###################


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

rps = train_datagen.flow_from_directory(
    'd:/study_data/_data/image/rps/',
    target_size=(100,100),
    batch_size=2520,
    class_mode='categorical',
    shuffle=False
) #Found 2520 images belonging to 3 classes.

x = rps[0][0]
y = rps[0][1]

print(x) 
print(y) 


print(x.shape,y.shape) #(2520, 100, 100, 3) (2520, 3)

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7,
                                                    shuffle=True
                                                    )



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

augument_size = 300 # 증폭
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
print(xy_train[0][1]) #(200,)
print(xy_train[0][1].shape) #(200,)

# np.save('d:/study_data/_save/_npy/keras49_8_train_x.npy', arr=xy_train[0][0])
# np.save('d:/study_data/_save/_npy/keras49_8_train_y.npy', arr=xy_train[0][1])
# np.save('d:/study_data/_save/_npy/keras49_8_test_x.npy', arr=x_test)
# np.save('d:/study_data/_save/_npy/keras49_8_test_y.npy', arr=y_test)
