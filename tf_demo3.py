import tensorflow as tf

sess = tf.InteractiveSession()
cons1 = tf.constant([6, 2, 5], shape=[4, 3])
# var1 = tf.Variable([0.8, 0.5], shape=[1, 2], name='weight')
var2 = tf.Variable(tf.random_normal(shape=[4,3],mean=0,stddev=1),name='v1')
a3 = tf.Variable(tf.ones(shape=[2,3]), name='a3')
sess.run(tf.global_variables_initializer())
print(sess.run(a3))
