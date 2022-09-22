import numpy as np
import tensorflow as tf

print(tf.__version__)
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if(gpus):
    print("쥐피유 돈다")
    aaa = 'gpu'
else: 
    print("쥐피유 안돈다") 
    aaa = 'cpu'   
