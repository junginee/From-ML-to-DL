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
# hypothseis는 input 값 x에 w를 곱한다.
# hypothesis는 y 에 해당되며 내가 추정한 값과 실제값의 오차가 가장 작을 때 가장 이상적

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))

#(내가 추정한 값 - 실제값)을 제곱한 후 합계에 대해 평균을 계산한 값이 loss이다.
#추정값 = 실제값이 가장 이상적인 상황이기에 cost가 0이 되는 지점을 학습으로 찾아가는 과정

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss) 
# 2차 함수 자체가 "loss"이다.
# loss가 가장 작은 값은 가장 아래 있는 붉은 점이다.
# 붉은점은 미분했을 때 기울기가 0인 유일한 지점이다.
# 즉, loss를 아래로 볼록한 2차 함수로 만들고,
# 미분했을 때 0인 지점의 w와 b값이 이상적인 결과이다.

#3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(loss), sess.run(w), sess.run(b))
        
        #loss는 0에 수렴해야한다.

sess.close()       