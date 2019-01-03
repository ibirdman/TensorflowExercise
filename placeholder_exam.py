import tensorflow as tf
import numpy as np

# 定义placeholder
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

# 定义乘法运算
output = tf.multiply(input1, input2)

# 通过session执行乘法运行
with tf.Session() as sess:
    # 执行时要传入placeholder的值
    print(sess.run(output, feed_dict = {input1:[7.], input2: [6.]}))

