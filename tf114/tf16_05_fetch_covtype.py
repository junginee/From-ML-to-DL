import tensorflow as tf 
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error,accuracy_score
from tensorflow.keras.utils import to_categorical
import numpy as np 
tf.set_random_seed(66)

# 1.데이터
datasets = fetch_covtype()

x_data = datasets.data

y_data = datasets.target 
y_data = to_categorical(y_data)

x_train, x_test, y_train, y_test = train_test_split(x_data,y_data, train_size = 0.9, random_state=123, stratify = y_data)

# print(x_train.dtype,y_train.dtype)

x = tf.placeholder(tf.float32, shape=[None, x_data.shape[1]])
y = tf.placeholder(tf.float32, shape=[None, y_data.shape[1]])

w = tf.Variable(tf.zeros([x_data.shape[1],y_data.shape[1]]), name = 'weights') 
b = tf.Variable(tf.zeros([1,y_data.shape[1]]), name='bias')

# 2.모델
h = tf.nn.softmax(tf.matmul(x, w) + b)  

# 3-1.컴파일 
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(h), axis=1))     # categorical_crossentropy
optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001)
train = optimizer.minimize(loss)

# 3-2.훈련 
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

epoch = 2000
for epochs in range(epoch):
    cost_val, hy_val, _  = sess.run([loss, h, train], feed_dict = {x:x_train, y:y_train})
    if epochs %20 == 0:

        print(epochs, 'loss : ', cost_val, 'hy_val : ', hy_val)


# #4.평가, 예측
y_predict = sess.run(tf.argmax(sess.run(h,feed_dict = {x:x_test}),axis=1))        
y_test = sess.run(tf.argmax(y_test,axis=1))         
                                                      
acc = accuracy_score(y_test,y_predict)
print('acc : ', acc)

sess.close()

# acc :  0.6849333929985199
