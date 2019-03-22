import tensorflow as tf
import numpy as np

#定义‘符号’变量，也称为占位符
a = tf.placeholder("float")
b = tf.placeholder("float")
y = tf.div(a, b) #构造一个op节点
sess = tf.Session()#建立会话
#运行会话，输入数据，并计算节点，同时打印结果
result = sess.run(y, feed_dict={a: 3, b: 3});
print("y=%d"%result);
# 任务完成, 关闭会话.
sess.close()
