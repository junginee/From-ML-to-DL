import numpy as np
from keras.preprocessing.image import ImageDataGenerator

#1. 데이터
train_datagen = ImageDataGenerator(
    rescale=1./255, #원본 영상은 0-255의 RGB 계수로 구성되는데, 이 같은 입력값은 모델을 효과적으로 학습시키기에 너무 높다. 
                    #그래서 이를 1/255로 스케일링하여 0-1 범위로 변환 /  이는 다른 전처리 과정에 앞서 가장 먼저 적용
           
    horizontal_flip=True, #수평방향으로 뒤집기
    vertical_flip=True, #수직방향으로 뒤집기
    width_shift_range=0.1, #지정된 수평방향 이동 범위내에서 임의로 원본이미지 이동
    height_shift_range=0.1, #지정된 수직방향 이동 범위내에서 임의로 원본이미지 이동
    rotation_range=5, #지정된 각도 범위내에서 임의로 원본이미지 회전
    zoom_range=1.2, #지정된 확대/축소 범위 내에서 임의로 원본이미지 확대/축소
    shear_range=0.7, #밀린 강도 범위내에서 임의로 원본이미지 변형
    # fill_mode = 'nearest' 이미지를 회전, 이동하거나 축소할 때 생기는 공간을 채우는 방식
    
)


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

# print(type(xy_train)) #<class 'keras.preprocessing.image.DirectoryIterator'>
# print(type(xy_train[0])) #<class 'tuple'>
# print(type(xy_train[0][0])) #<class 'numpy.ndarray'>
# print(type(xy_train[0][1])) #<class 'numpy.ndarray'>

#현재 (5,200,200,1) 데이터가 32 덩어리

'''
#2. 모델
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(32,(2,2), input_shape= (200,200,1), activation='relu'))
model.add(Conv2D(32,(2,2), activation='relu'))
model.add(Flatten())
model.add(Dense(13, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3.컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#model.fit(xy_train[0][1], xy_train[0][1]) #만약 배치를 최대로 잡았다면 이 표현도 가능
hist = model.fit_generator(xy_train, epochs=10, steps_per_epoch=32,
                                         # 통상적으로 steps_per_epoch는 '전체데이터/batch = 160/5 = 32'와 같이 기재
                    validation_data= xy_test,
                    validation_steps=4)

accuracy = hist.history['accuracy']
val_accuracy = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss : ',loss[-1])
print('val_loss :', val_loss[-1])
print('accuracy : ',accuracy[-1])
print('val_accuracy :',val_accuracy[-1])

import matplotlib.pyplot as plt

plt.figure(figsize=(9,6)) #그래프 표 사이즈
plt.plot(hist.history['loss'], marker = '.' ,c = 'red', label = 'loss') # maker: 점으로 표시하겟다  ,c:색깔 ,label : 이름
plt.plot(hist.history['val_loss'], marker = '.' ,c = 'blue', label = 'val_loss')
plt.grid() # 모눈종이에 하겠다
plt.title('keras47_2_fit_generator')#제목
plt.ylabel('loss')#y축 이름
plt.xlabel('epochs')#x축 이름
plt.legend(loc='upper right') # upper right: 위쪽 상단에 표시하겠다.(라벨 이름들)
plt.show()# 보여줘

# loss :  0.6931734681129456
# val_loss : 0.6934290528297424
# accuracy :  0.5
# val_accuracy : 0.44999998807907104

'''
