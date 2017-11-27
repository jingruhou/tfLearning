#coding:utf-8
import tensorflow as tf

hello = tf.constant('Hello, TensorFlow')
sess =tf.Session()
result = sess.run(hello)
print result

a = tf.constant(10)
b = tf.constant(32)
print sess.run(a + b)

sess.close()