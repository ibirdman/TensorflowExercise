import tensorflow as tf
import numpy as np

c = np.array([[3.,5], [5.,6], [6.,7]])
print(np.mean(c,1))

Mean = tf.reduce_mean(c,1)
with tf.Session() as sess:
    result = sess.run(Mean)
    print(result)
