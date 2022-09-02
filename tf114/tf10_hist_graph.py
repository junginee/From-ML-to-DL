import tensorflow as tf
tf.set_random_seed(123)

#1. 데이터
x_train_data = [1, 2, 3]
y_train_data = [3, 5, 7]
x_train = tf.placeholder(tf.float32, shape=[None])
y_train = tf.placeholder(tf.float32, shape=[None])
w = tf.Variable(tf.random_normal([1]), dtype=tf.float32)
b = tf.Variable(tf.random_normal([1]), dtype=tf.float32)

#2. 모델구성
hypothesis = x_train * w + b

#3-1. 컴파일
loss = tf.reduce_mean(tf.square(hypothesis - y_train))


optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.172)
train = optimizer.minimize(loss) 

#3-2. 훈련
loss_val_list = [ ]
w_val_list=[ ]
with tf.compat.v1.Session() as sess : 

    sess.run(tf.global_variables_initializer()) 
    epochs =101
    for step in range(epochs):
        # sess.run(train)
        _, loss_val, w_val, b_val = sess.run([train, loss, w, b],
                      feed_dict = {x_train: x_train_data, y_train:y_train_data})

  
        if step % 2== 0:
            print(step, loss_val, w_val, b_val) 
        
        loss_val_list.append(loss_val)
        w_val_list.append(w_val)
        
    x_test_data = [6, 7, 8]
    x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])

    y_predict = x_test *w_val + b_val  #y_predict = model.predict(x_test)
    print("[6,7,8] 예측 :",   sess.run(y_predict, feed_dict={x_test : x_test_data})  )
    
    
import matplotlib.pyplot as plt

plt.subplot(2,1,1)
plt.plot(loss_val_list, marker = 'o')
plt.xlabel('epochs')
plt.ylabel('loss')


plt.subplot(2,1,2)
plt.plot(w_val_list, marker='+')
plt.xlabel('epochs')
plt.ylabel('w_val_loss')

plt.show()
