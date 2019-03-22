import tensorflow as tf
import numpy as np

'''
x = tf.constant(
[[1., 1.],
[2., 2.]]);

a = tf.reduce_mean(x, axis=1);

sess = tf.Session()
with sess.as_default():
    print(a.eval());'''



t = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
b = tf.reshape(t, [2, -1]);
sess = tf.Session()
with sess.as_default():
    print(b.eval());