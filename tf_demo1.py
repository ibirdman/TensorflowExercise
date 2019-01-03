import tensorflow as tf

# 首先，创建一个TensorFlow常量=>2
const = tf.constant(3.0, name='const')

# 创建TensorFlow变量b和c
b = tf.Variable(2.0, name='b')
c = tf.Variable(1.0, dtype=tf.float32, name='c')

# 创建operation
d = tf.add(b, c, name='add1')
e = tf.add(c, const, name='add2')
a = tf.multiply(d, e, name='multiply')

init_op = tf.global_variables_initializer()

# session
with tf.Session() as sess:
    # 2. 运行init operation
    sess.run(init_op)
    # 计算
    result = sess.run(a)
    print("Variable a is {}".format(result))
    
writer = tf.summary.FileWriter("logs/demo1", tf.get_default_graph())
writer.close()
