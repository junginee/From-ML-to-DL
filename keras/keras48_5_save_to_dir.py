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

augument_size = 20
randidx = np.random.randint(x_train.shape[0], size=augument_size) 
# 정수 임의 추출함수 : randint
# 0~59999 범위 내에서 40000개 정수를 추출한다.

print(randidx)
print(np.min(randidx), np.max(randidx))
print(type(randidx)) #<class 'numpy.ndarray'>

x_augumented = x_train[randidx].copy()
y_augumented = y_train[randidx].copy() 

print(x_augumented.shape) #(40000, 28, 28)
print(y_augumented.shape) #(40000,)

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
                                  save_to_dir= 'D:\study_data\_temp',
                              shuffle=False).next()[0]  #증폭한 x 데이터를 train_datagen에 넣는다.

end_time = time.time()-start_time    
print(augument_size, '개 증폭에 걸린시간 :', round(end_time,3), "초")

# print(x_augumented)
# print(x_augumented.shape) #(40000, 28, 28, 1)

# x_train = np.concatenate((x_train, x_augumented))
# y_train = np.concatenate((y_train, y_augumented))

# print(x_train.shape, y_train.shape) #(100000, 28, 28, 1) (100000,)


# print(x_train[0].shape) #(28, 28)
# print(x_train[0].reshape(28*28).shape) #(784,)
# print(np.tile(x_train[0].reshape(28*28), augument_size).reshape(-1,28,28,1).shape) 
# #np.tile=배열 복사하기, 붙여넣기 함수 (100, 28, 28, 1)

# print(np.zeros(augument_size))
# print(np.zeros(augument_size).shape) #(100,)

# x_data = train_datagen.flow(
#     np.tile(x_train[0].reshape(28*28), augument_size).reshape(-1,28,28,1),
#     np.zeros(augument_size),
#     batch_size=augument_size,
#     shuffle=True).next()  # batch_size만큼의 랜덤하게 변형된 학습 데이터를 만든다.

 
# print(x_data) #<keras.preprocessing.image.NumpyArrayIterator object at 0x000002A794291D90>


######################## .next() 사용 ####################################
# print(x_data[0]) #batch만큼 자른 x, y데이터 출력         
# print(x_data[0].shape) #(100, 28, 28, 1)    .next() 사용해 x 출력
# print(x_data[1].shape) #(100,)              .next() 사용해 y 출력


# ######################## .next() 미사용 ###################################
# print(x_data[0])          #batch만큼 자른 x, y데이터 출력   
# print(x_data[0][0].shape) #(100, 28, 28, 1)   .next() 미사용 x 출력
# print(x_data[0][1].shape) #(100,)             .next() 미사용 y 출력

# import matplotlib.pyplot as plt
# plt.figure(figsize=(10,10))
# for i in range(100):
#   plt.subplot(10,10,i+1)
#   plt.axis('off')
#   plt.imshow(x_data[0][0][i], cmap='winter_r')     
# plt.show()    
