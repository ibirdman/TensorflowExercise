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
tf.app.flags.DEFINE_integer("iteration", 10001, "Iterations to train [1e4]")
tf.app.flags.DEFINE_integer("input_num", 2, "the num of input class")
tf.app.flags.DEFINE_integer("hide1_num", 5, "the node num of layer1")
tf.app.flags.DEFINE_integer("hide2_num", 8, "the node num of layer2")
tf.app.flags.DEFINE_integer("output_num", 0, "the num of output class") # got dynamically
tf.app.flags.DEFINE_integer("train_batch_size", 5, "The size of batch images [128]")
tf.app.flags.DEFINE_integer("disp_freq", 500, "Display the current results every display_freq iterations [1e2]")
tf.app.flags.DEFINE_float("learning_rate", 0.1, "Learning rate of for adam [0.01]")
tf.app.flags.DEFINE_string("log_dir", "logs", "Directory of logs.")


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

    # 2 隐藏层2
    with tf.name_scope('Hide2'):
        W = tf.Variable(initial_value=tf.truncated_normal(shape=[FLAGS.hide1_num, FLAGS.hide2_num], stddev=0.01), name='Weights')
        b = tf.Variable(initial_value=tf.zeros(shape=[FLAGS.hide2_num]), name='bias')
        hide2 = tf.nn.relu((tf.matmul(hide1, W) + b))

    # 3 输出层
    with tf.name_scope('Output'):
        W = tf.Variable(initial_value=tf.truncated_normal(shape=[FLAGS.hide2_num, FLAGS.output_num], stddev=0.01), name='Weights')
        b = tf.Variable(initial_value=tf.zeros(shape=[FLAGS.output_num]), name='bias')
        output = tf.matmul(hide2, W) + b
        Y_pred = tf.nn.softmax(logits=output)

    logits = output
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits, name='cross_entropy')
    loss = tf.reduce_mean(cross_entropy, name='loss')

    optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(loss)

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
        x_pred_map = create_predict_map(all_images)
        # print(x_pred_map)
        y_pred_map = sess.run(Y_pred, feed_dict={X: x_pred_map})
        # print(np.argmax(y_pred_map, 1))
        draw_pred_map(x_pred_map, y_pred_map)
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

