import tensorflow as tf
from sklearn.datasets import load_breast_cancer
tf.compat.v1.set_random_seed(66)

# 1. 데이터

datasets = load_breast_cancer()
x_data = datasets.data
y_data = datasets.target

print(x_data.shape, y_data.shape)  #(569, 30) (569,)
y_data = y_data.reshape(-1,1)      #(569, 30) (569,1)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data,y_data,
        train_size = 0.7,random_state=9)


# 2. 모델구성
x = tf.compat.v1.placeholder(tf.float32, shape = [None, 30] )
y = tf.compat.v1.placeholder(tf.float32, shape = [None, 1] )

# w = tf.Variable(tf.random.normal([30,1])) #값이 엄청 낮고 nan값 출력 
# b = tf.Variable(tf.random.normal([1])) #0.64 수준 값은 출력된다

# 초기값을 설정하는 것에서 왜 정규분포를 따르는
# random.normal() 사용하면 nan 출력?  
w = tf.Variable(tf.zeros([30,1]),name='weight')
b = tf.Variable(tf.zeros([1]), name='bias')


# hypothesis 에서 값을 제한한다 ! 
hypothesis = tf.sigmoid(tf.matmul(x, w) + b) # 값을 0-1 사이의 값을 내보낸다 
cost = -tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))
#바이너리 크로스 엔트로피 
#결과값이 마이너스로 나오기 때문에 - 붙여 줘야한다 

# 3-1. 컴파일
loss =   - tf.reduce_mean(y*tf.log(hypothesis)+(1-y)*tf.log(1-hypothesis))

optimizer = tf.train.AdamOptimizer(learning_rate=0.0000011)
train = optimizer.minimize(cost)

# 3-2. 훈련
sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epochs in range(2001):
    loss_val, hy_val, _ = sess.run([loss, hypothesis, train], feed_dict={x:x_data, y:y_data})
    
    if epochs % 10 == 0:
        print(epochs, 'loss :',loss_val, '\n', hy_val)

#평가 예측
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
#조건에 대해서 true = 1, false = 0 출력 
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, y ), dtype=tf.float32))


c, a = sess.run([predicted, accuracy], feed_dict={x:x_test, y:y_test})
print("예측값 : \n", hy_val, 
        "\n predict : \n", c, 
        "\n Accuracy :", a)

from sklearn.metrics import r2_score, accuracy_score
# accs = accuracy_score(y_test, predicted)
# print(accs)

sess.close()