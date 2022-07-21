### 모델구성 ###
# 성능비교, 증폭 전 후 비교

from tensorflow.keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import  Conv2D, MaxPool2D, Flatten, Dense, Dropout
import numpy as np
import pandas as pd


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

augument_size = 40000
randidx = np.random.randint(x_train.shape[0], size=augument_size) 
# 정수 임의 추출함수 : randint
# 0~59999 범위 내에서 40000개 정수를 추출한다.
# x_train.shape(60000, 28, 28)
#              [0]    [1]  [2]     

print(randidx.shape)
print(np.min(randidx), np.max(randidx))
print(type(randidx)) #<class 'numpy.ndarray'>

x_augumented = x_train[randidx].copy()
y_augumented = y_train[randidx].copy() 

print(x_augumented.shape) #(40000, 28, 28)
print(y_augumented.shape) #(40000,)

x_train = x_train.reshape(60000,28,28,1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

print(x_train.shape)
print(x_test.shape)

    
x_augumented = x_augumented.reshape(x_augumented.shape[0], 
                                    x_augumented.shape[1], 
                                    x_augumented.shape[2], 1)

x_augumented = train_datagen.flow(x_augumented, y_augumented, #형식상 x,y 둘다 넣어야 한다.
                                  batch_size=augument_size,
                                  shuffle=False).next()[0]  #증폭한 x 데이터를 train_datagen에 넣는다.
                                                            #상단에서 random 추출했으므로 shuffle은 false 
print(x_augumented)
print(x_augumented.shape) #(40000, 28, 28, 1)

x_train = np.concatenate((x_train, x_augumented))
y_train = np.concatenate((y_train, y_augumented))

print(x_train.shape, y_train.shape) #(100000, 28, 28, 1) (100000,)

#--------------------------------------------------------------------
#y에 대한 전처리(원핫인코딩)

y_train = pd.get_dummies(y_train)
y_test = pd.get_dummies(y_test)

#2. 모델 구성
model = Sequential()
model.add(Conv2D(10, kernel_size=(3,3), strides=2, padding='same', input_shape=(28, 28, 1)))   
model.add(MaxPool2D())
model.add(Conv2D(5, (3,3), activation='relu'))  
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16))
model.add(Dense(10, activation='softmax'))

#3. 컴파일
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', restore_best_weights=True)

model.fit(x_train, y_train, epochs=5, batch_size=2, validation_split=0.2, callbacks=[es])


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss : ' , loss[0])

print("----------------------------------------")

from tensorflow.keras.utils import to_categorical 
y_predict = model.predict(x_test)
y_predict = np.argmax(y_predict) #열(axis= 1)에서 최댓값을 구한다.
y_test = np.argmax(y_test)

from sklearn.metrics import  accuracy_score  
acc= accuracy_score(y_test, y_predict)
print('acc스코어 : ', acc)




