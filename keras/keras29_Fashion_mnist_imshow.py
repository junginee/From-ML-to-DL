import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train),(x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape, y_train.shape) 
print(x_test.shape, y_test.shape)   

print(x_train[0])
print(y_train[0])

import matplotlib.pyplot as plt
plt.imshow(x_train[7], 'black')
plt.show()
