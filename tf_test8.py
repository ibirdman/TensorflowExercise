import numpy as np
import tensorflow as tf


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


labels = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
logits = np.array([[11., 8., 7.], [10., 14., 3.], [1., 2., 4.]])
y_pred = sigmoid(logits)
print(y_pred)
prob_error1 = -labels * np.log(y_pred) - (1 - labels) * np.log(1 - y_pred)
print(prob_error1)

print(".............")
labels1 = np.array([[0., 1., 0.], [1., 1., 0.], [0., 0., 1.]])  # 不一定只属于一个类别
logits1 = np.array([[1., 8., 7.], [10., 14., 3.], [1., 2., 4.]])
y_pred1 = sigmoid(logits1)
prob_error11 = -labels1 * np.log(y_pred1) - (1 - labels1) * np.log(1 - y_pred1)
print(prob_error11)

print(".............")
with tf.Session() as sess:
    print(sess.run(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)))
    print(".............")
    print(sess.run(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels1, logits=logits1)))