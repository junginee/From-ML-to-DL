import tensorflow as tf
tf.compat.v1.set_random_seed(66)

#1. 데이터
x = [1,2,3]
y = [3,5,7]
x_test_d = [5,6,7]

W = tf.Variable(tf.random_normal([1],dtype=tf.float32))   
b = tf.Variable(tf.random_normal([1],dtype=tf.float32))    

x_train = tf.placeholder(tf.float32, shape=[None])  
y_train = tf.placeholder(tf.float32, shape=[None])  
x_test = tf.placeholder(tf.float32, shape=[None])  

#2. 모델구성
hypothesis = x * W + b

### 1. Session() // sess.run

loss = tf.reduce_mean(tf.square(hypothesis - y_train))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.16)
train = optimizer.minimize(loss)  

# with tf.compat.v1.Session() as sess:
#     sess.run(tf.compat.v1.global_variables_initializer())

#     for step in range(100):
#      # sess.run(train)
#         _, loss_val, w_val, b_val = sess.run([train, loss, W, b], 
#                                         feed_dict={x_train:x, y_train:y})

#     #4. 예측
#     predict = x_test * w_val + b_val 

#     print("1. [6,7,8] 예측 : " , sess.run(predict, feed_dict={x_test:x_test_d}))


### 2. Session() // eval 

# sess = tf.compat.v1.Session()
# sess.run(tf.compat.v1.global_variables_initializer())

# for step in range(100):
#     _, loss_val, w_val, b_val = sess.run([train, loss, W, b], 
#                                         feed_dict={x_train:x, y_train:y})


# predict = x_test * w_val + b_val     

# print("2. [6,7,8] 예측 : " , predict.eval(session=sess, feed_dict={x_test:x_test_d}))

# sess.close()


###3. InteractiveSession() // eval 

sess = tf.compat.v1.InteractiveSession()
sess.run(tf.compat.v1.global_variables_initializer())

for step in range(100):
    # sess.run(train)
    _, loss_val, w_val, b_val = sess.run([train, loss, W, b],
                                        feed_dict={x_train:x, y_train:y})

predict = x_test * w_val + b_val  

print("3. [6,7,8] 예측 : " , predict.eval(feed_dict={x_test:x_test_d}))

sess.close()        
