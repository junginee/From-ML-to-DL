import numpy as np
from keras.datasets import mnist
                                                                    
(x_train, _), (x_test, _) = mnist.load_data()   

x_train = x_train.reshape(60000, 784).astype('float32')/255.               
x_test = x_test.reshape(10000, 784).astype('float32')/255.

from keras.models import Sequential, Model                            
from keras.layers import Dense, Input                   

input_img = Input(shape=(784,))

# encoded = Dense(1064, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(input_img)
# encoded = Dense(16, activation='relu')(input_img)             
# encoded = Dense(1, activation='relu')(input_img)

decoded = Dense(784, activation='sigmoid')(encoded)
# decoded = Dense(784, activation='relu')(encoded)
# decoded = Dense(784, activation='linear')(encoded)
# decoded = Dense(784, activation='tanh')(encoded)

autoencoder = Model(input_img, decoded)

# autoencoder.summary()

autoencoder.compile(optimizer='adam', loss = 'binary_crossentropy',
                                    metrics=['acc'])

autoencoder.fit(x_train, x_train, epochs=30, batch_size=128,
                           validation_split=0.2)

decoded_imgs = autoencoder.predict(x_test)

import matplotlib.pyplot as plt
n = 10
plt.figure(figsize=(20,4)) 
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    
    ax = plt.subplot(2, n, i+1+n)
    plt.imshow( decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
