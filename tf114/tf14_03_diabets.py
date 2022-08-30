import tensorflow as tf 
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
datasets = load_diabetes()
x_data, y_data= datasets.data,datasets.target

tf.compat.v1.set_random_seed(123)

y_data = y_data.reshape(-1,1)

x_train, x_test, y_train, y_test = train_test_split(x_data,y_data, train_size=0.9, random_state=123)

x = tf.compat.v1.placeholder(tf.float32, shape=[None,x_data.shape[1]])
y = tf.compat.v1.placeholder(tf.float32, shape=[None,1])

w = tf.compat.v1.Variable(tf.compat.v1.random_normal([x_data.shape[1],1],dtype=tf.float32), name = 'weights') #행렬곱 shape 을 맞춰줘야함 y(h)가 (5, 1)이므로 w(3,1)을 곱해줘야함
b = tf.compat.v1.Variable(tf.compat.v1.random_normal([1],dtype=tf.float32), name='bias')

# 2.모델
h = tf.compat.v1.matmul(x, w) + b  

# 3-1.컴파일

loss = tf.reduce_mean(tf.square(h - y))
optimizer = tf.train.AdamOptimizer(learning_rate = 1) 
train = optimizer.minimize(loss)

# 3-2.훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epoch = 2000
for epochs in range(epoch):
    cost_val, hy_val, _  = sess.run([loss, h, train],
                                    feed_dict = {x:x_train, y:y_train})
    if epochs %20 == 0:
        print(epochs, 'loss : ', cost_val)


# 4.평가, 예측
from sklearn.metrics import r2_score,mean_absolute_error

y_predict = sess.run(h, feed_dict={x:x_test})

r2 = r2_score(y_test,y_predict)
print('r2 : ', r2)

mae = mean_absolute_error(y_train,hy_val)
print('mae : ', mae)

sess.close()


# r2 :  0.639865686270985
# mae :  43.964366038440154