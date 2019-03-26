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

SAMPLE_DATA_FILE = '../data/mysamples3.csv';

# 定义一个全局对象来获取参数的值，在程序中使用(eg：FLAGS.iteration)来引用参数
FLAGS = tf.app.flags.FLAGS

# 设置训练相关参数
tf.app.flags.DEFINE_integer("iteration", 50001, "Iterations to train [1e4]")
tf.app.flags.DEFINE_integer("input_num", 2, "the num of input class")
tf.app.flags.DEFINE_integer("hide1_num", 5, "the node num of layer1")
tf.app.flags.DEFINE_integer("hide2_num", 8, "the node num of layer2")
tf.app.flags.DEFINE_integer("output_num", 0, "the num of output class") # got dynamically
tf.app.flags.DEFINE_integer("train_batch_size", 5, "The size of batch images [128]")
tf.app.flags.DEFINE_integer("disp_freq", 500, "Display the current results every display_freq iterations [1e2]")
tf.app.flags.DEFINE_float("learning_rate", 0.1, "Learning rate of for adam [0.01]")
tf.app.flags.DEFINE_string("log_dir", "logs", "Directory of logs.")
tf.app.flags.DEFINE_float("lambd", 0.008, "regularization weight")


def main(argv=None):
    # 0、准备训练/验证/测试数据集
    sample_set = sm.load_samples(SAMPLE_DATA_FILE, one_hot=True)
    labels = np.vstack((sample_set.train.labels, sample_set.validation.labels, sample_set.test.labels))
    # calculate the output class num
    labels = np.argmax(labels, 1);
    FLAGS.output_num = max(labels) + 1

    # 1 输入层
    with tf.name_scope('Input'):
        X = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.input_num], name='X_placeholder')
        Y = tf.placeholder(dtype=tf.int32, shape=[None, FLAGS.output_num], name='Y_placeholder')

    # 2 隐藏层1
    with tf.name_scope('Hide1'):
        W = tf.Variable(initial_value=tf.truncated_normal(shape=[FLAGS.input_num, FLAGS.hide1_num], stddev=0.01), name='Weights')
        b = tf.Variable(initial_value=tf.zeros(shape=[FLAGS.hide1_num]), name='bias')
        hide1 = tf.nn.relu((tf.matmul(X, W) + b))
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(FLAGS.lambd)(W))

    # 2 隐藏层2
    with tf.name_scope('Hide2'):
        W = tf.Variable(initial_value=tf.truncated_normal(shape=[FLAGS.hide1_num, FLAGS.hide2_num], stddev=0.01), name='Weights')
        b = tf.Variable(initial_value=tf.zeros(shape=[FLAGS.hide2_num]), name='bias')
        hide2 = tf.nn.relu((tf.matmul(hide1, W) + b))
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(FLAGS.lambd)(W))

    # 3 输出层
    with tf.name_scope('Output'):
        W = tf.Variable(initial_value=tf.truncated_normal(shape=[FLAGS.hide2_num, FLAGS.output_num], stddev=0.01), name='Weights')
        b = tf.Variable(initial_value=tf.zeros(shape=[FLAGS.output_num]), name='bias')
        output = tf.matmul(hide2, W) + b
        Y_pred = tf.nn.softmax(logits=output)
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(FLAGS.lambd)(W))

    logits = output
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=logits, name='cross_entropy')
    loss = tf.reduce_mean(cross_entropy, name='loss')
    tf.add_to_collection('losses', loss)
    total_loss = tf.add_n(tf.get_collection('losses'))

    optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(total_loss)
    # optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(total_loss)

    correct_prediction = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print('~~~~~~~~~~~开始执行计算图~~~~~~~~~~~~~~')
    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(logdir=FLAGS.log_dir, graph=sess.graph)
        # 初始化所有变量
        sess.run(tf.global_variables_initializer())
        total_loss = 0
        for i in range(0, FLAGS.iteration):
            X_batch, Y_batch = sample_set.train.next_batch(FLAGS.train_batch_size)
            _, loss_batch = sess.run([optimizer, loss], feed_dict={X: X_batch, Y: Y_batch})
            total_loss += loss_batch
            if i % FLAGS.disp_freq == 0:
                val_acc = sess.run(accuracy, feed_dict={X: sample_set.validation.images, Y: sample_set.validation.labels})
                if i == 0:
                    print('step: {}, train_loss: {}, val_acc: {}'.format(i, total_loss, val_acc))
                else:
                    print('step: {}, train_loss: {}, val_acc: {}'.format(i, total_loss / FLAGS.disp_freq, val_acc))
                total_loss = 0

        test_acc = sess.run(accuracy, feed_dict={X: sample_set.test.images, Y: sample_set.test.labels})
        print('test accuracy: {}'.format(test_acc))
        summary_writer.close()
        # print(sess.run(W))

        # redraw
        plt.clf()
        all_images = np.vstack((sample_set.train.images, sample_set.validation.images, sample_set.test.images))
        all_labels = np.vstack((sample_set.train.labels, sample_set.validation.labels, sample_set.test.labels))
        draw_train_data(all_images, all_labels)
        plt.legend(loc='lower right')
        xx, yy = create_predict_map(all_images)
        x_pred_map = np.c_[xx.ravel(), yy.ravel()]
        y_pred_map = sess.run(Y_pred, feed_dict={X: x_pred_map})
        draw_pred_map(x_pred_map, y_pred_map)
        draw_pred_contour(xx, yy, y_pred_map)
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title('sample points')
        plt.show()

def init_figure():
    fig = plt.figure(figsize=(10, 8))

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
        plt.scatter(X[index, 0], X[index, 1], marker=marker, color=color, label=label, s=15, alpha=0.2)

# draw split line
def draw_pred_contour(xx, yy, pred_map):
    grid = np.c_[xx.ravel(), yy.ravel()]
    pred = np.argmax(pred_map, 1)
    zz = pred.reshape(xx.shape)
    plt.contour(xx, yy, zz, 1)

# draw split map
def create_predict_map(X):
    num = 100
    xmin = min(X[:, 0])
    xmax = max(X[:, 0])
    xunit = (xmax - xmin) / num
    # print(xmin, xmax + xunit, xunit)
    ymin = min(X[:, 1])
    ymax = max(X[:, 1])
    yunit = (ymax - ymin) / num
    # print(ymin, ymax + yunit, yunit)
    xx, yy = np.mgrid[xmin:xmax + xunit:xunit, ymin:ymax + yunit:yunit]
    return xx, yy

if __name__ == '__main__':
    init_figure()
    tf.app.run()

