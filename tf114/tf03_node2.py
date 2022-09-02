import tensorflow as tf

node1 = tf.constant(2.0, tf.float32)
node2 = tf.constant(3.0, tf.float32)
node3 = tf.add(node1, node2) # node3 = node1 + node2
node4 = tf.subtract(node1, node2)
node5 = tf.multiply(node1, node2)
node6 = tf.div(node1, node2)


sess = tf.Session()
print(sess.run(node3)) 
print()
print(sess.run(node4)) 
print()
print(sess.run(node5)) 
print()
print(sess.run(node6))

import tensorflow as tf
node1 = tf.constant(1,2,3)
node2 = tf.constant(4,5,6)
tf.print(node1, node2)
