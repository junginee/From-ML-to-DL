import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from keras.applications import VGG16

#model = VGG16()  #include_top=True, input_shape=(224, 224, 3)
vgg16 = VGG16(weights='imagenet', include_top=False,
               input_shape=(32,32,3))

# [1]
vgg16.trainable=False #가중치 동결 
# vgg16.summary()

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(10))

# [2]
model.trainable=False

model.summary()

                                                                #trainable:True / VGG False / model False
print(len(model.weights))                        #            30               30             30
print(len(model.trainable_weights))        #            30               4                0
                                                                #   [1],[2]주석O        [2]주석      [1], [2]주석 x

