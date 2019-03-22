import tensorflow as tf
import numpy as np

#-------------------1. 数据集，变量，占位符------------------------#

# 样本，输入列表，正太分布(Normal Destribution)，均值为1, 均方误差为0.1, 数据量为100个
x_vals = np.random.normal(1, 0.1, 100)
# 样本, 输出列表， 100个值为10.0的列表
y_vals = np.repeat(10.0, 100)

#占位符
x_data = tf.placeholder(shape=[None, 1], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype= tf.float32)

#模型变量
A = tf.Variable(tf.random_normal(shape=[1, 1]))

#批量大小
batch_size = 25

#训练数据集的index，从总样本的index，即0～99,选取80个值
train_indices = np.random.choice(len(x_vals), round(len(x_vals) *0.8), replace = False)
#测试数据集的index，扣除上面的train_indices，剩下的20个值
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))

#训练数据集 & 测试数据集
x_vals_train = x_vals[train_indices]
x_vals_test = x_vals[test_indices]
y_vals_train = y_vals[train_indices]
y_vals_test = y_vals[test_indices]

#-----------------2. 模型，损失函数，优化器算法--------------------------#

# 我们定义的模型，是一个线型函数，即 y = w * x， 也就是my_output = A * x_data
# x_data将用样本x_vals。我们的目标是，算出A的值。
# 其实已经能猜出，y都是10.0的话，x均值为1, 那么A应该是10。哈哈
my_output = tf.multiply(x_data, A)

# 损失函数， 用的是模型算的值，减去实际值， 的平方。y_target就是上面的y_vals。
loss = tf.reduce_mean(tf.square(my_output - y_target))

#初始化变量
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

# 梯度下降算法， 学习率0.02, 可以认为每次迭代修改A，修改一次0.02。比如A初始化为20, 发现不好，于是猜测下一个A为20-0.02
my_opt = tf.train.GradientDescentOptimizer(learning_rate=0.02)
train_step = my_opt.minimize(loss)#目标，使得损失函数达到最小值


#-----------------3. 迭代训练--------------------------#


for i in range(1000):#0到100,不包括100
    # 随机拿25个index
    rand_index = np.random.choice(len(x_vals_train), size = batch_size)
    # 从训练集拿出25个样本，转置一下，因为x_data的shape是[None, 1]
    #注意是[x_vals_train[rand_index]]，转为二维的1x20的数组，才能通过transpose转置为20x1的数组，不能写成x_vals_train[rand_index]
    rand_x = np.transpose([x_vals_train[rand_index]])
    rand_y = np.transpose([y_vals_train[rand_index]])
    #损失函数引用的placeholder(直接或间接用的都算), x_data使用样本rand_x， y_target用样本rand_y
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    #打印
    if i%25==0:
        print('step: ' + str(i) + ' A = ' + str(sess.run(A)))
        print('loss: ' + str(sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})))



#-----------------4. 评估模型--------------------------#
#以上这种评估，测试集跟训练集是完全分开的。没有用A去评测测试集，只看两种集的均方误差是不是差不多
mse_test = sess.run(loss, feed_dict={x_data: np.transpose([x_vals_test]), y_target: np.transpose([y_vals_test])})
mse_train = sess.run(loss, feed_dict={x_data: np.transpose([x_vals_train]), y_target: np.transpose([y_vals_train])})
print('MSE on test: ' + str(np.round(mse_test, 2)))
print('MSE on train: ' + str(np.round(mse_train, 2)))

