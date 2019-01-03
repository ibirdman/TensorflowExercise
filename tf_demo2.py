import tensorflow as tf

x = tf.constant(1.0, name='input')
w = tf.Variable(0.8, name='weight')
y = tf.multiply(w, x, name='output')
y_ = tf.constant(0.5, name='correct_value')
loss = tf.pow(y - y_, 2, name='loss')

train_step = tf.train.GradientDescentOptimizer(0.025).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

for value in [x, w, y, y_, loss]:
    tf.summary.scalar(value.op.name, value)

summaries = tf.summary.merge_all()

with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter('log_simple_stats', sess.graph)
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        summary_writer.add_summary(sess.run(summaries), i)
        sess.run(train_step)
        print(sess.run(w))
        print(sess.run(loss))
