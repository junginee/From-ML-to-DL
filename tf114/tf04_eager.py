#즉시 실행모드
# tf.compat.v1.disable_eager_execution()
#즉시실행모드 기재 시 >> 2~ 버전에서 실행
#즉시실행모드 미기재 시 >> 1~ 버전에서 실행

import tensorflow as tf
print(tf.__version__)
print(tf.executing_eagerly()) #True 현재 스레드가 열망 실행을 활성화 한 경우 true
print(tf.executing_eagerly())
      
hello = tf.constant("Hello world")

sess = tf.compat.v1.Session()
print(sess.run(hello)) #b'Hello world'          
