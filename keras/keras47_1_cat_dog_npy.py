import numpy as np
from keras.preprocessing.image import ImageDataGenerator

#1. 데이터    

x_train = np.load('d:/study_data/_save/_npy/keras47_1_train_x.npy')
y_train = np.load('d:/study_data/_save/_npy/keras47_1_train_y.npy')
x_test = np.load('d:/study_data/_save/_npy/keras47_1_test_x.npy')
y_test = np.load('d:/study_data/_save/_npy/keras47_1_test_y.npy')


#2. 모델
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(32,(2,2), input_shape= (150,150,3), activation='relu'))
model.add(Conv2D(32,(2,2), activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#3.컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=50, validation_split=0.2) 


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)  

from sklearn.metrics import accuracy_score
y_predict = y_predict.round()
acc = accuracy_score(y_test, y_predict)

print('loss : ', round(loss[0],2))
print('acc : ', acc)


# loss :  3.35
# acc :  0.521
