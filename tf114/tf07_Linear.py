# y = wx + b

import tensorflow as tf
tf.set_random_seed(123)

#1. 데이터
x = [1,2,3]
y = [1,2,3]

w = tf.Variable(11, dtype=tf.float32)
b = tf.Variable(10, dtype=tf.float32)

#2. 모델구성
hypothesis = x * w + b

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss) 

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(loss), sess.run(w), sess.run(b))


sess.close()       
