import tensorflow as tf 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,mean_squared_error
import numpy as np 
tf.set_random_seed(66)

# 1.데이터
datasets = load_iris()

x_data = datasets.data
y_data = datasets.target 

from tensorflow.keras.utils import to_categorical
y_data = to_categorical(y_data)


x_train, x_test, y_train, y_test = train_test_split(x_data,y_data, train_size = 0.9, random_state=123, stratify = y_data)


# print(x_train.dtype,y_train.dtype)

x = tf.placeholder(tf.float32, shape=[None, x_data.shape[1]])
y = tf.placeholder(tf.float32, shape=[None, y_data.shape[1]])

w = tf.Variable(tf.random_normal([x_data.shape[1],y_data.shape[1]]), name = 'weights') 
b = tf.Variable(tf.random_normal([1,y_data.shape[1]]), name='bias')

# 2.모델
h = tf.nn.softmax(tf.matmul(x, w) + b)  

# 3-1.컴파일 
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(h), axis=1))     # categorical_crossentropy
optimizer = tf.train.AdamOptimizer(learning_rate = 0.1)
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
y_predict = sess.run(h,feed_dict={x:x_test})
y_predict = sess.run(tf.argmax(y_predict,axis=1))           #텐서를 새로운 형태로 캐스팅하는데 사용한다.부동소수점형에서 정수형으로 바꾼 경우 소수점 버림.
y_test = sess.run(tf.argmax(y_test,axis=1))             #텐서를 새로운 형태로 캐스팅하는데 사용한다.부동소수점형에서 정수형으로 바꾼 경우 소수점 버림.
                                                                       #Boolean형태인 경우 True이면 1, False이면 0을 출력한다.

from sklearn.metrics import r2_score,mean_absolute_error,accuracy_score

acc = accuracy_score(y_test,y_predict)
print('acc : ', acc)

sess.close()

# acc :  0.8666666666666667
