import tensorflow  as tf
import numpy as np

 

x_data = [[1,2,1,1], 
                [2,1,3,2],  
                [3,1,3,4],  
                [4,1,5,5], 
                [1,7,5,5], 
                [1,2,5,6], 
                [1,6,6,6], 
                [1,7,6,7]]

y_data = [[0,0,1], 
                [0,0,1], 
                [0,0,1], 
                [0,1,0], 
                [0,1,0], 
                [0,1,0], 
                [1,0,0], 
                [1,0,0]]

 
x = tf. compat. v1.placeholder(tf.float32, shape = [None,4]) 
w = tf.Variable(tf.random.normal([4, 3]), name='weight')
b = tf.Variable(tf.random.normal([1, 3]), name='bias')  

y = tf.compat.v1.placeholder(tf.float32, shape = [None, 3] )

# BASIC : hypothesis = x * w + b
# NeuralNetwork
hypothesis = tf.nn.softmax(tf.matmul(x, w) + b) 
# model.add(Dense(3, activation='softmax'))

# 3-1. 컴파일
########### Categorical_Crossentropy ###########
loss = tf.reduce_mean( - tf.reduce_sum(y * tf.log(hypothesis), axis=1)  )
# loss = 'categorical_crossentropy'

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

# train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(2001):
        _, loss_val = sess.run([optimizer, loss], feed_dict={x:x_data, y:y_data})
        if step % 200 == 0:
            print(step, loss_val )
            
    # Predict
    results = sess.run(hypothesis, feed_dict={x:x_predict} )
    print(results, sess.run(tf.arg_max(results, 1)))
