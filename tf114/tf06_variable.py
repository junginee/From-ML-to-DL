import tensorflow as tf
sess = tf.compat.v1.Session()

x = tf.Variable([2], dtype=tf.float32)
y = tf.Variable([3], dtype=tf.float32)

init = tf.compat.v1.global_variables_initializer()
sess.run(init)
#변수는 위의 내장함수로 초기화 시킨다.

print(sess.run(x+y))
