# 넘파이에서 불러와서 모델 구성
# 성능 비교 

# 증폭해서 npy에 저장
from click import argument
from keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

#1. 데이터 로드

x_train = np.load('d:/study_data/_save/_npy/keras49_1_train_x.npy')
y_train = np.load('d:/study_data/_save/_npy/keras49_1_train_y.npy')
x_test = np.load('d:/study_data/_save/_npy/keras49_1_test_x.npy')
y_test = np.load('d:/study_data/_save/_npy/keras49_1_test_y.npy')


#2. 모델
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Conv2D, Dropout, Dense, MaxPool2D, Flatten

model = Sequential()
model.add(Conv2D(10, kernel_size=(3,3), strides=2, padding='same', input_shape=(28, 28, 1)))   # 27, 27, 6  
model.add(MaxPool2D())
model.add(Conv2D(5, (3,3), activation='relu'))   # 7, 7, 5
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16))
model.add(Dense(10, activation='softmax'))


#3. 컴파일, 훈련
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['mse','accuracy'])

import time
start_time = time.time()
# hist = model.fit_generator(xy_train, epochs=100, steps_per_epoch=len(xy_train))
hist = model.fit(x_train, y_train, epochs=100, batch_size=32,
                 validation_split=0.2)
end_time = time.time() - start_time

accuracy = hist.history['accuracy']
val_accuracy = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss :', loss[-1])       
print('val_loss :', val_loss[-1])
print('accuracy :', accuracy[-1])
print('val_accuracy :', val_accuracy[-1])

print("=====================================================================")
print("걸린시간 : ", end_time)


#그래프로 비교
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

font_path = 'C:\Windows\Fonts\malgun.ttf'
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.plot(hist.history['accuracy'], marker='.', c='orange', label='accuracy')
plt.plot(hist.history['val_accuracy'], marker='.', c='green', label='val_accuracy')
plt.grid()    
plt.title('로스값과 검증로스값')    
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend()   
plt.show()


#=============================== 이전 결과값 ====================================#
# loss :  0.4753
# accuracy :  0.8275
#================================================================================#


#=============================== 증폭 후 결과값 ===================================#
# loss : 0.4843232333660126
# val_loss : 0.7431361079216003
# accuracy : 0.8171499967575073
# val_accuracy : 0.7240999937057495
# =====================================================================
# 걸린시간 :  991.4976036548615
#================================================================================#