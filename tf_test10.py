import tensorflow as tf
import numpy as np

labels = np.array([[0., 1., 4.], [1, 0, 0]])
logits = [[2., 1., 3.], [1, 2, 1]]
with tf.Session() as sess:
    a = sess.run(tf.argmax(labels,1))
    b = sess.run(tf.argmax(logits,1))
    correct_prediction = tf.equal(tf.argmax(labels, 1), tf.argmax(logits, 1))  # 判断预测标签和实际标签是否匹配
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    c = sess.run(correct_prediction)
    d = sess.run(tf.cast(correct_prediction, "float"))
    e = sess.run(accuracy)
    print(a, b, c, d, e)