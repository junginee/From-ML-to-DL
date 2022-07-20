

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from regex import X
from sklearn.model_selection import train_test_split

#1. 데이터    

x = np.load('d:/study_data/_save/_npy/keras47_2_x.npy')
y = np.load('d:/study_data/_save/_npy/keras47_2_y.npy')


x_train, x_test, y_train, y_test = train_test_split(x,y,
                                                    train_size=0.8,
                                                    random_state=66
                                                    )

#2. 모델
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(32,(2,2), input_shape= (100,100,3), activation='relu'))
model.add(Conv2D(32,(2,2), activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(5))
model.add(Dense(1, activation='sigmoid'))

#3.컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=100, validation_split=0.2) 


#4. 평가, 예측

loss = model.evaluate(x_test, y_test)
y_predict = model.predict(x_test)  

from sklearn.metrics import accuracy_score
y_predict = y_predict.round()
acc = accuracy_score(y_test, y_predict)

print('loss : ', round(loss[0],2))
print('acc : ', round(acc,2))

# loss :  1.13
# acc :  0.76



