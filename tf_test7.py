import tensorflow as tf

labels = [0., 1., 0.]
logits = [2., 10., 3.]
with tf.Session() as sess:
    a = sess.run(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels,logits=logits))
    print(a)