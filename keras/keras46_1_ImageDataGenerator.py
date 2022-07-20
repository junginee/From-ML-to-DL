import numpy as np
from keras.preprocessing.image import ImageDataGenerator

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
    'd:/_data/image/brain/train/',
    target_size=(150,150),
    batch_size=5,
    class_mode = 'binary',
    color_mode = 'grayscale',
    shuffle=True,
)  #Found 160 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
    'd:/_data/image/brain/test/',
    target_size=(150,150), #이미지 사이즈 동일하지 않기 때문에 일괄적으로 150,150으로 고정
    batch_size=5,
    class_mode = 'binary',
    color_mode = 'grayscale',
    shuffle=True,
)  #Found 120 images belonging to 2 classesv.


print(xy_train[0])
print(xy_train[31][0]) #뒤에 [0]로 하면 x값
print(xy_train[31][0]) #뒤에 [1]로 하면 y값
# print(xy_train[31][2]) #error

print(xy_train[2][0].shape)  #(5, 150, 150, 1)
print(xy_train[2][1].shape)  #(5,)

print(type(xy_train)) #<class 'keras.preprocessing.image.DirectoryIterator'>
print(type(xy_train[0])) #<class 'tuple'>
print(type(xy_train[0][0])) #<class 'numpy.ndarray'> #image 데이터를 가져왔을 때 x numpy, y numpy 형태로 batch 단위로 묶여있다.
print(type(xy_train[0][1])) #<class 'numpy.ndarray'> #image 데이터를 가져왔을 때 x numpy, y numpy 형태로 batch 단위로 묶여있다.
