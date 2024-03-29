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

#2. 모델구성
hypothesis = x_train * w + b

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y_train))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.055)
train = optimizer.minimize(loss) 

#3-2. 훈련
with tf.compat.v1.Session() as sess : 
#sess = tf.compat.v1.Session() 
    sess.run(tf.global_variables_initializer()) #세션 열기
    epochs =101
    for step in range(epochs):
         _, loss_val, w_val, b_val = sess.run([train, loss, w, b],
                      feed_dict = {x_train: x_train_data, y_train:y_train_data})
  
        if step % 2== 0:
            print(step, loss_val, w_val, b_val) 
        
    x_test_data = [6, 7, 8]
    x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])

    y_predict = x_test *w_val + b_val  #y_predict = model.predict(x_test)
    print("[6,7,8] 예측 :",   sess.run(y_predict, feed_dict={x_test : x_test_data})  )
    
    
#  98 0.037762474 [2.2227132] [0.49372053]
# [6,7,8] 예측 : [13.819025 16.038794 18.258562]
