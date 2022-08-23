# 실습
#lr 수정해서 epoch 100번 이하로 줄이기
#step = 100 이하, w=1.99, b=0.99


import tensorflow as tf
tf.set_random_seed(123)

#1. 데이터
x_train_data = [1, 2, 3]
y_train_data = [3,5,7]

x_train = tf.placeholder(tf.float32, shape=[None]) #레이어마다 shape 명시
y_train = tf.placeholder(tf.float32, shape=[None])


# w = tf.Variable(111, dtype=tf.float32)
# b = tf.Variable(10, dtype=tf.float32)
w = tf.Variable(tf.random_normal([1]), dtype=tf.float32) #[숫자] = 출력되는 개수 
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32) #[숫자] = 출력되는 개수

# sess = tf.compat.v1.Session() #세션 열기
# sess.run(tf.global_variables_initializer()) #초기화
# print(sess.run(w)) #런


#2. 모델구성
hypothesis = x_train * w + b
# 선형회귀
# hypothseis는 input 값 x에 w를 곱한다.
# hypothesis는 y 에 해당되며 내가 추정한 값과 실제값의 오차가 가장 작을 때 가장 이상적

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y_train))

#(내가 추정한 값 - 실제값)을 제곱해준 것이 loss이다.
#추정값 = 실제값이 가장 이상적인 상황이기에 cost가 0이 되는 지점을 학습으로 찾아가는 과정

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.055)
train = optimizer.minimize(loss) 
#loss를 아래로 볼록한 2차 함수로 만들고,
#미분했을 때 0인 지점의 w와 b값이 이상적인 결과이다.

#3-2. 훈련
with tf.compat.v1.Session() as sess : 
#sess = tf.compat.v1.Session() 
    sess.run(tf.global_variables_initializer()) #세션 열기
    epochs =101
    for step in range(epochs):
        # sess.run(train)
        _, loss_val, w_val, b_val = sess.run([train, loss, w, b],
                      feed_dict = {x_train: x_train_data, y_train:y_train_data})
        #_ = 반환하지 않지만 실행하겠다.
  
        if step % 2== 0:
            print(step, loss_val, w_val, b_val) #with문이 끝나는 시점에 자동으로 close
        
    x_test_data = [6, 7, 8]
    x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])

    y_predict = x_test *w_val + b_val  #y_predict = model.predict(x_test)
    print("[6,7,8] 예측 :",   sess.run(y_predict, feed_dict={x_test : x_test_data})  )
    
    
#  98 0.037762474 [2.2227132] [0.49372053]
# [6,7,8] 예측 : [13.819025 16.038794 18.258562]