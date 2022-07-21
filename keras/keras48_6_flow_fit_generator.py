from tensorflow.keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import  Conv2D, MaxPool2D, Flatten, Dense, Dropout
import numpy as np
import pandas as pd

(x_train, y_train),(x_test, y_test) = fashion_mnist.load_data()



#-----------------augument 데이터 만들기---------------------------------------------------

augument_size = 40000
batch_size=64
randidx = np.random.randint(x_train.shape[0], size=augument_size) #x_train.shape의 첫번째 shape에 위치한 데이터 중 40000개 랜덤정수 추출

x_augumented = x_train[randidx].copy()  #(40000, 28, 28)
y_augumented = y_train[randidx].copy()  #(40000,)

x_augumented = x_augumented.reshape(x_augumented.shape[0],x_augumented.shape[1], 
                                    x_augumented.shape[2], 1)
print(x_augumented.shape) #(40000, 28, 28, 1)

train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=0.1,
    shear_range=0.7,
    fill_mode='nearest')

x_augumented = train_datagen.flow(x_augumented, y_augumented,
                                  batch_size=augument_size,
                                  shuffle=False).next()[0] #x_augumented만 증폭 시킨 후 저장 
#---------------------------------------------------------------------------------------




#---x_train과 x_test reshape 후 concatenate 사용해서 증폭된 데이터 + 기존 데이터 합치기-----

x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)


x_train= np.concatenate((x_train, x_augumented)) #(100000, 28, 28, 1)
y_train = np.concatenate((y_train, y_augumented)) #(100000)

#---------------------------------------------------------------------------------------


#---x_train과 y_train을 xy_train으로 ImageDataGenerator한다.------------------------------

train_datagen2 = ImageDataGenerator(rescale=1./255)
 
xy_train = train_datagen2.flow(x_train, y_train, 
                               batch_size=64,
                               shuffle=False)  

#---------------------------------------------------------------------------------------

#2. 모델 구성
model = Sequential()
model.add(Conv2D(10, kernel_size=(3,3), strides=2, padding='same', input_shape=(28, 28, 1)))   
model.add(MaxPool2D())
model.add(Conv2D(5, (3,3), activation='relu'))  
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16))
model.add(Dense(10, activation='softmax'))



#3. 컴파일
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', restore_best_weights=True)

hist=model.fit_generator(xy_train, epochs=1)




#4. 평가, 예측
accuracy=hist.history['accuracy']
loss = hist.hisory['loss']

print('loss : ', loss[-1])
print('accuracy : ', loss[-1])




