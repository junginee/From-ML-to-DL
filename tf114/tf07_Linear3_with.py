# y = wx + b

import tensorflow as tf
tf.set_random_seed(123)

#1. 데이터
x = [1, 2, 3, 4, 5]
y = [1, 2, 3, 4, 5]

w = tf.Variable(333, dtype=tf.float32)
b = tf.Variable(5, dtype=tf.float32)

#2. 모델구성
hypothesis = x * w + b
# hypothseis는 input 값 x에 w를 곱한다.
# hypothesis는 y 에 해당되며 내가 추정한 값과 실제값의 오차가 가장 작을 때 가장 이상적

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y))

#(내가 추정한 값 - 실제값)을 제곱해준 것이 loss이다.
#추정값 = 실제값이 가장 이상적인 상황이기에 cost가 0이 되는 지점을 학습으로 찾아가는 과정

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss) 
#loss를 아래로 볼록한 2차 함수로 만들고,
#미분했을 때 0인 지점의 w와 b값이 이상적인 결과이다.

#3-2. 훈련
with tf.compat.v1.Session as sess : 
#sess = tf.compat.v1.Session() #세션 열기
    sess.run(tf.global_variables_initializer())

    epochs =5000
    for step in range(epochs):
        sess.run(train)
        if step % 500 == 0:
            print(step, sess.run(loss), sess.run(w), sess.run(b)) #with문이 끝나는 시점에 자동으로 close
        
#sess.close()   #세션 닫기