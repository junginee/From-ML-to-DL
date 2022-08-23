# y = wx + b

import tensorflow as tf
tf.set_random_seed(123)

#1. 데이터
# x = [1, 2, 3, 4, 5]
# y = [1, 2, 3, 4, 5]

x = tf.placeholder(tf.float32, shape=[None]) #레이어마다 shape 명시
y = tf.placeholder(tf.float32, shape=[None])


# w = tf.Variable(111, dtype=tf.float32)
# b = tf.Variable(10, dtype=tf.float32)
w = tf.Variable(tf.random_normal([1]), dtype=tf.float32) #[숫자] = 출력되는 개수 
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32) #[숫자] = 출력되는 개수

# sess = tf.compat.v1.Session() #세션 열기
# sess.run(tf.global_variables_initializer()) #초기화
# print(sess.run(w)) #런


#2. 모델구성
hypothesis = x * w + b
# 선형회귀
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
with tf.compat.v1.Session() as sess : 
#sess = tf.compat.v1.Session() #세션 열기
    sess.run(tf.global_variables_initializer())

    epochs =2001
    for step in range(epochs):
        # sess.run(train)
        _, loss_val, w_val, b_val = sess.run([train, loss, w, b],
                      feed_dict = {x :[1,2,3,4,5], y:[1,2,3,4,5]})
        #_ = 반환하지 않지만 실행하겠다.
  
        if step % 20 == 0:
            print(step, loss_val, w_val, b_val) #with문이 끝나는 시점에 자동으로 close
        
#sess.close()   #세션 닫기
