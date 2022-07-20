

import numpy as np
from keras.preprocessing.image import ImageDataGenerator

#1. 데이터    


# np.save('d:/study_data/_save/_npy/keras46_5_train_x.npy', arr=xy_train[0][0])
# np.save('d:/study_data/_save/_npy/keras46_5_train_x.npy', arr=xy_train[0][1])
# np.save('d:/study_data/_save/_npy/keras46_5_test_x.npy', arr=xy_test[0][0])
# np.save('d:/study_data/_save/_npy/keras46_5_test_x.npy', arr=xy_test[0][1])

x_train = np.load('D:\study_data\_save\_npy\keras46_5_train_x.npy')
y_train = np.load('d:/study_data/_save/_npy/keras46_5_train_y.npy')
x_test = np.load('d:/study_data/_save/_npy/keras46_5_test_x.npy')
y_test = np.load('d:/study_data/_save/_npy/keras46_5_test_y.npy')


print(x_train.shape) #(160, 150, 150, 1)
print(y_train.shape) #(160,)
print(x_test.shape)  #(120, 150, 150, 1)
print(y_test.shape)  #(120,)


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

#model.fit(xy_train[0][0], xy_train[0][1]) #만약 배치를 최대로 잡았다면 이 표현도 가능
hist = model.fit_generator(x_train, y_train, epochs=10, steps_per_epoch=32,
                                         # 통상적으로 steps_per_epoch는 '전체데이터/batch = 160/5 = 32'와 같이 기재
                          validation_steps=4)

accuracy = hist.history['accuracy']
val_accuracy = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

print('loss : ',loss[-1])
print('val_loss :', val_loss[-1])
print('accuracy : ',accuracy[-1])
print('val_accuracy :',val_accuracy[-1])



