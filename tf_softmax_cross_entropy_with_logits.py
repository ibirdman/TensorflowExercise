import tensorflow as tf
import numpy as np;

labels = np.array([0., 1., 0.])
logits = np.array([2., 6., 3.])

def softmax(y):
    sum = np.sum(np.exp(y))
    y = np.exp(y) / sum
    return y

def cross_entropy(labels, logits):
    error = -labels * np.log(logits) - (1 - labels) * np.log(1 - logits)
    return error


a = softmax(logits)
print(a)
b = cross_entropy(labels, a)
print(b)
print(np.max(b))

with tf.Session() as sess:
    a = sess.run(tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logits))
    print(a)

with tf.Session() as sess:
    a = sess.run(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=1,logits=logits))
    print(a)