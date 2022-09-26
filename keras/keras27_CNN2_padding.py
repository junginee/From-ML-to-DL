from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
model = Sequential()
model.add(Conv2D(filters=10, kernel_size = (3,3), 
                 padding='same', 
                 input_shape = (28,28,1) ))
model.add(MaxPooling2D())
model.add(Conv2D(32,(2,2), 
                 padding = 'valid',  
                 activation = 'relu'))   

model.add(Flatten()) #출력 : (N, 28)
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary() 



'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 28, 28, 10)        100
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 14, 14, 10)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 13, 13, 32)        1312
_________________________________________________________________
flatten (Flatten)            (None, 5408)              0
_________________________________________________________________
dense (Dense)                (None, 32)                173088
_________________________________________________________________
dense_1 (Dense)              (None, 32)                1056
_________________________________________________________________
dense_2 (Dense)              (None, 10)                330
=================================================================
Total params: 175,886
Trainable params: 175,886
Non-trainable params: 0
_________________________________________________________________


'''
