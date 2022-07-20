import numpy as np 
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rescale=1./255)

xy = datagen.flow_from_directory(
    'D:\study_data\_data\image\men-or-women', # 이 경로의 이미지파일을 불러 수치화
    target_size=(150,150),# 크기들을 일정하게 맞춰준다.
    batch_size=9000,
    class_mode='binary', 
    # color_mode='grayscale', #디폴트값은 컬러
    shuffle=True,
    )

# print(xy_train)
print(xy)

np.save('D:\study_data\_save\_npy\keras47_04_x.npy', arr=xy[0][0])
np.save('D:\study_data\_save\_npy\keras47_04_y.npy', arr=xy[0][1])