# import tensorflow as tf
# print(tf.__version__)

# # print("hello world")        

# hello = tf.constant("hello world")           
# # print(hello)

# sess = tf.compat.v1.Session()
# print(sess.run(hello))
   
import tensorflow.compat.v1 as tf

# import numpy as np

 

hello = tf.constant("hello world")

sess=tf.Session()

print(sess.run(hello))
