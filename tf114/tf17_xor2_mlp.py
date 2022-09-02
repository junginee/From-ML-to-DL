import tensorflow as tf
tf.compat.v1.set_random_seed

#1. 데이터
x_data = [[0,0], [0,1], [1,0], [1,1]]    #(4, 2)
y_data = [[0], [1], [1], [0]]               #(4, 1)

#2. 모델구성

#input layer
x = tf.compat.v1.placeholder(tf.float32, shape=[None, 2])
y = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

#hidden layer
w1 = tf.Variable(tf.random.normal([2, 20]), name='weight')
b1 = tf.Variable(tf.random.normal([20]), name='bias')
hidden_layer1 = tf.matmul(x, w1) + b1       

#output layer
w2 = tf.Variable(tf.random.normal([20, 1]), name='weight2')
b2 = tf.Variable(tf.random.normal([1]), name='bias2')
hypothesis = tf.matmul(hidden_layer1, w2) + b2       


# 3-1. 컴파일
loss =   - tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))   
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001)
train = optimizer.minimize(loss)
x_test = tf.compat.v1.placeholder(tf.float32, shape=[None])

# 3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epochs in range(1000):
      
    loss_val, hy_val, _ = sess.run([loss, hypothesis, train], 
                          feed_dict={x:x_data, y:y_data})
    
    if epochs % 20 == 0:
        print(epochs, 'loss :',loss_val, '\n', hy_val)


# 4. 평가, 예측
y_predict = tf.cast( hypothesis > 0.5, dtype=tf.float32 )

print( sess.run( hypothesis>0.5, feed_dict = {x:x_data, y:y_data} ) )
print( sess.run(tf.equal(y,y_predict), feed_dict={x:x_data, y:y_data}))

accuracy = tf.reduce_mean( tf.cast(tf.equal(y, y_predict), dtype=tf.float32) )
pred, acc = sess.run([y_predict, accuracy],feed_dict={x: x_data, y:y_data})

print("=========================================================")
print("예측값 : \n", hy_val)
print("예측결과 : \n", pred)
print("Accuracy : ", acc)

sess.close()