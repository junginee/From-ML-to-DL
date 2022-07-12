import numpy as np
from keras.datasets import cifar10
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler


#1. 데이터
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print(x_train.shape, y_train.shape) #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape) #(10000, 32, 32, 3) (10000, 1)


#--------------------------------------------------------------------
# x에 대한 전처리

scaler = StandardScaler()
x_train = x_train.reshape(50000,-1)
x_test = x_test.reshape(10000,-1)

print(x_train.shape, x_test.shape) #(50000, 3072) (10000, 3072)

scaler.fit(x_train)
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#x_train = x_train.reshape(50000, -1)  
#x_test = x_test.reshape(10000, -1)

#print(x_train.shape, x_test.shape) #(50000, 3072) (10000, 3072)

#--------------------------------------------------------------------
# y에 대한 전처리(원핫인코딩)

import numpy as np
print(np.unique(y_train)) #[0 1 2 3 4 5 6 7 8 9]

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)



#2.모델구성
model = Sequential()
model.add(Dense(32, activation='relu', input_dim=3072))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax')) 
model.summary()

#3.컴파일 훈련

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

from tensorflow.python.keras.callbacks import EarlyStopping



earlyStopping =EarlyStopping(monitor='val_loss', patience=5, mode='min', verbose=1, 
                             restore_best_weights=True) 



hist = model.fit(x_train, y_train, epochs=10, batch_size=50, 
                validation_split=0.2,
                callbacks=[earlyStopping], # 최저값을 체크해 반환해줌
                verbose=1)



#4.평가 예측
loss = model.evaluate(x_test, y_test) 
print('loss : ', loss[0])
print('accuracy : ', loss[1])

y_predict = model.predict(x_test)

#print(y_test)
#print(y_predict)
y_predict = np.argmax(y_predict, axis= 1)
y_test = np.argmax(y_test, axis= 1)

from sklearn.metrics import accuracy_score
#acc = accuracy_score(y_test, y_predict)
#print(acc)

# loss :  1.4884288311004639
# accuracy :  0.46810001134872437
