import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
train_datagen = ImageDataGenerator(
    rescale=1./255, #원본 영상은 0-255의 RGB 계수로 구성되는데, 이 같은 입력값은 모델을 효과적으로 학습시키기에 너무 높다. 
                    #그래서 이를 1/255로 스케일링하여 0-1 범위로 변환 /  이는 다른 전처리 과정에 앞서 가장 먼저 적용
           
    horizontal_flip=True, 
    vertical_flip=True, 
    width_shift_range=0.1, 
    height_shift_range=0.1, 
    rotation_range=5,
    zoom_range=1.2, 
    shear_range=0.7,
    fill_mode = 'nearest')


test_datagen = ImageDataGenerator(
    rescale = 1./255
    
)

data = train_datagen.flow_from_directory(
    'D:\study_data\_data\image\horse-or-human',
    target_size=(100,100),
    batch_size=10000,
    class_mode = 'binary',
    #color_mode = 'grayscale',
    shuffle=True
) #Found 1027 images belonging to 2 classes.


print(data[0][0].shape) #(1027, 100, 100, 3)
print(data[0][1].shape) #(1027,)


np.save('d:/study_data/_save/_npy/keras47_2_x.npy', arr=data[0][0])
np.save('d:/study_data/_save/_npy/keras47_2_y.npy', arr=data[0][1])
