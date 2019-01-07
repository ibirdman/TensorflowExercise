import tensorflow as tf
import os
import sample_manager as sm
import matplotlib.pyplot as plt
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# 定义一个全局对象来获取参数的值，在程序中使用(eg：FLAGS.iteration)来引用参数
FLAGS = tf.app.flags.FLAGS

# 设置训练相关参数
tf.app.flags.DEFINE_integer("iteration", 101, "Iterations to train [1e4]")
tf.app.flags.DEFINE_integer("disp_freq", 200, "Display the current results every display_freq iterations [1e2]")
tf.app.flags.DEFINE_integer("train_batch_size", 5, "The size of batch images [128]")
tf.app.flags.DEFINE_float("learning_rate", 0.1, "Learning rate of for adam [0.01]")
tf.app.flags.DEFINE_string("log_dir", "logs", "Directory of logs.")

def main(argv=None):
    # 0、准备训练/验证/测试数据集
    # mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    sample_set = sm.load_samples('data/mysamples3.csv')

    # 1、数据输入设计：使用 placeholder 将数据送入网络，None 表示张量的第一个维度可以是任意长度的
    with tf.name_scope('Input'):
        X = tf.placeholder(dtype=tf.float32, shape=[None, 2], name='X_placeholder')
        Y = tf.placeholder(dtype=tf.int32, shape=[None, 1], name='Y_placeholder')

    # 2、前向网络设计
    with tf.name_scope('Inference'):
        print(sample_set.train.images)
        W = tf.Variable(initial_value=tf.random_normal(shape=[2, 3], stddev=0.01), name='Weights')
        b = tf.Variable(initial_value=tf.zeros(shape=[3]), name='bias')
        logits = tf.matmul(X, W) + b
        Y_pred = tf.nn.softmax(logits=logits)

    # 3、损失函数设计
    with tf.name_scope('Loss'):
        # 求交叉熵损失
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits, name='cross_entropy')
        # 求平均
        loss = tf.reduce_mean(cross_entropy, name='loss')

    # 4、参数学习算法设计
    with tf.name_scope('Optimization'):
        optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(loss)

    # 5、评估节点设计
    with tf.name_scope('Evaluate'):
        # 返回验证集/测试集预测正确或错误的布尔值
        correct_prediction = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(Y, 1))
        # 将布尔值转换为浮点数后，求平均准确率
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
                    print('step: {}, train_loss: {}, val_acc: {}'.format(i, total_loss/FLAGS.disp_freq, val_acc))
                total_loss = 0

        test_acc = sess.run(accuracy, feed_dict={X: sample_set.test.images, Y: sample_set.test.labels})
        print('test accuracy: {}'.format(test_acc))
        summary_writer.close()
        # redraw
        plt.clf()
        draw_train_data(sample_set.train.images, sample_set.train.labels)
        draw_predict_map(sess)
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.legend(loc = 'upper left')
        plt.title('samples points') 
        plt.show()
        
def init_figure():
    fig = plt.figure()
    fig.canvas.mpl_connect('key_press_event', on_key_press)    
    
def on_key_press(event): 
    if event.key == 'ctrl+z':
        redraw_all()
        
def draw_train_data(X, Y):
    index_0 = np.where(Y==0)[0]
    plt.scatter(X[index_0,0], X[index_0,1], marker='x', color = 'b', label = '0', s = 50)
    index_1 =np.where(Y==1)[0]
    plt.scatter(X[index_1,0], X[index_1,1], marker='o', color = 'r', label = '1', s = 15)
    index_2 =np.where(Y==2)[0]
    plt.scatter(X[index_2,0], X[index_2,1], marker='v', color = 'g', label = '2', s = 15)
    
def draw_predict_map(sess):
    X1 = np.linspace(min(X[:,0]), max(X[:,0]), num=20)
    X2 = np.linspace(min(X[:,1]), max(X[:,1]), num=20)
    for x1 in X1:
        for x2 in X2:
           test_sample = np.array([1, x1, x2], dtype=np.float)
           y_predict = logistic_classify.predict(test_sample)
           plt.scatter(x1, x2, c = np.where(y_predict == 0, 'blue', 'red'), alpha = 0.2, edgecolors='none', s=10)
           
def redraw(X, Y):
    plt.clf()

    draw_train_data(X, Y)
    # draw_predict_map(X)

    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend(loc = 'upper left')
    plt.title('samples points') 
    plt.show()           


# 执行main函数
if __name__ == '__main__':
    tf.app.run()
    init_figure()
