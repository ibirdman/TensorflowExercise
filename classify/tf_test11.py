import tensorflow as tf
import os
import sys

sys.path.append("..")
import sample.sample_manager as sm
import matplotlib.pyplot as plt
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
OUTPUT_CLASS_STYLES = np.array([
    ['0', 'o', 'r'],
    ['1', 'x', 'g'],
    ['2', 'v', 'b'],
    ['3', 's', 'k'],
    ['4', '*', 'y'],
    ['5', '^', '#808000'],
])

SAMPLE_DATA_FILE = '../data/mysamples4.csv';

# 定义一个全局对象来获取参数的值，在程序中使用(eg：FLAGS.iteration)来引用参数
FLAGS = tf.app.flags.FLAGS

# 设置训练相关参数
tf.app.flags.DEFINE_integer("iteration", 1001, "Iterations to train [1e4]")
tf.app.flags.DEFINE_integer("output_num", 0, "the num of output class")
tf.app.flags.DEFINE_integer("train_batch_size", 5, "The size of batch images [128]")
tf.app.flags.DEFINE_integer("disp_freq", 200, "Display the current results every display_freq iterations [1e2]")
tf.app.flags.DEFINE_float("learning_rate", 0.1, "Learning rate of for adam [0.01]")
tf.app.flags.DEFINE_string("log_dir", "logs", "Directory of logs.")


"""
权重初始化
初始化为一个接近0的很小的正数
"""
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)


def main(argv=None):
    # 0、准备训练/验证/测试数据集
    sample_set = sm.load_samples(SAMPLE_DATA_FILE, one_hot=True)
    labels = np.vstack((sample_set.train.labels, sample_set.validation.labels, sample_set.test.labels))
    labels = np.argmax(labels, 1)
    FLAGS.output_num = max(labels) + 1


def init_figure():
    fig = plt.figure(figsize=(10, 8))
    fig.canvas.mpl_connect('key_press_event', on_key_press)


def on_key_press(event):
    if event.key == 'ctrl+z':
        redraw_all()


def draw_train_data(X, Y):
    Y = np.argmax(Y, 1);
    for i in range(FLAGS.output_num):
        index = np.where(Y == i)[0]
        label = OUTPUT_CLASS_STYLES[i, 0]
        marker = OUTPUT_CLASS_STYLES[i, 1]
        color = OUTPUT_CLASS_STYLES[i, 2]
        plt.scatter(X[index, 0], X[index, 1], marker=marker, color=color, label=label, s=30)


def draw_pred_map(X, Y):
    Y = np.argmax(Y, 1);
    for i in range(FLAGS.output_num):
        index = np.where(Y == i)[0]
        label = OUTPUT_CLASS_STYLES[i, 0]
        marker = OUTPUT_CLASS_STYLES[i, 1]
        color = OUTPUT_CLASS_STYLES[i, 2]
        plt.scatter(X[index, 0], X[index, 1], marker=marker, color=color, label=label, s=15, alpha=0.1)


def create_predict_map(X):
    x1_num = 100
    x2_num = 100
    x_map = np.zeros(shape=(x1_num * x2_num, 2), dtype=np.float)
    X1 = np.linspace(min(X[:, 0]), max(X[:, 0]), num=x1_num)
    X2 = np.linspace(min(X[:, 1]), max(X[:, 1]), num=x2_num)
    num = 0
    for x1 in X1:
        for x2 in X2:
            x_map[num, 0] = x1
            x_map[num, 1] = x2
            num += 1
    return x_map


# 执行main函数
if __name__ == '__main__':
    init_figure()
    tf.app.run()

