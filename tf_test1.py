import tensorflow as tf

a = tf.constant([1, 2], name="a")
b = tf.constant([3, 5], name="b")
result = a + b

sess = tf.Session();
out = sess.run(result)
print("The result is {}.".format(out))

writer = tf.summary.FileWriter("logs/bbb", tf.get_default_graph())
writer.close()
