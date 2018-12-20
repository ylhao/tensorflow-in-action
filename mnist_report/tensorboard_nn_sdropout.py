# encoding: utf-8


import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import shutil
from sklearn.utils import shuffle


batch_size = 64
max_step = 1000
learning_rate = 0.001
data_dir = './tmp/data/mnist/input_data'  # 数据
log_dir = './nn_tmp/logs/mnist_with_summaries'  # 日志
ckpt_dir = './nn_tmp/ckpt'  # 模型


mnist = input_data.read_data_sets(data_dir, one_hot=True)
X_temp = mnist.train.images
y_temp = mnist.train.labels
X_temp, y_temp = shuffle(X_temp, y_temp)
X_train = X_temp[0:50000]
y_train = y_temp[0:50000]
X_valid = X_temp[50000:55000]
y_valid = y_temp[50000:55000]
X_test = mnist.test.images
y_test = mnist.test.labels


# check
print('X_temp shape:', X_temp.shape)
print('y_temp shape:', y_temp.shape)
print('X_train shape:', X_train.shape)
print('y_train shape:', y_train.shape)
print('X_valid shape:', X_valid.shape)
print('y_valid shape:', y_valid.shape)
print('X_test shape:', X_test.shape)
print('y_test shape:', y_test.shape)


start_idx = 0
def next_batch(batch_size=batch_size):
    global start_idx
    idx = start_idx
    if idx + batch_size > X_train.shape[0]:
        start_idx = 0
        return X_train[idx: ], y_train[idx: ]
    else:
        start_idx += batch_size
        return X_train[idx: idx + batch_size], y_train[idx: idx + batch_size]


def safe_mkdir(path):
    """
    如果目录不存在，创建目录
    """
    if not os.path.exists(path):
        os.makedirs(path)


def clean_dir(path):
    """
    清空文件夹
    """
    if os.path.exists(path):
        shutil.rmtree(path)
    safe_mkdir(path)


def weight_variable(shape):
    """
    初始化权重矩阵
    """
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """
    初始化偏置
    """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def variable_summary(var):
    """
    variable summary
    """
    with tf.name_scope('summaries'):
        with tf.name_scope('mean'):  # 计算均值
            mean = tf.reduce_mean(var)
        with tf.name_scope('stddev'):  # 计算标准差
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        with tf.name_scope('max'):  # 计算最大值
            max = tf.reduce_max(var)
        with tf.name_scope('min'):  # 计算最小值
            min = tf.reduce_min(var)
        tf.summary.scalar('mean', mean)  # 均值
        tf.summary.scalar('stddev', stddev)  # 标准差
        tf.summary.scalar('max', max)  # 最大值
        tf.summary.scalar('min', min)  # 最小值
        tf.summary.histogram('histogram', var)  # 直方图


def dense_layer(input, input_dim, output_dim, layer_name='dense-layer', act=tf.nn.relu):
    """
    add a dense layer
    """
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):  # 该层权重
            weights = weight_variable([input_dim, output_dim])
            variable_summary(weights)
        with tf.name_scope('biases'):  # 该层偏置
            biases = bias_variable([output_dim])
            variable_summary(biases)
        with tf.name_scope('Wx_plus_b'):  # 线性变换
            preactivate = tf.matmul(input, weights) + biases
            tf.summary.histogram('preactivations', preactivate)  # 线性变换结果的直方图
        activations = act(preactivate, name='activation')  # 非线性变换
        tf.summary.histogram('activation', activations)  # 非线性变换结果的直方图
        return activations  # 返回非线性变换的结果


def conv_layer(input, filter=[5, 5, 1, 32], strides=[1, 1, 1, 1], padding='SAME', act=tf.nn.relu, layer_name='conv-layer'):
    """
    add a convolutional layer
    filter: kernel size, eg: [5, 5, 1, 32]
    """
    filter_num = filter[-1]  # 获取卷积核数目
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = weight_variable(filter)
            variable_summary(weights)
        with tf.name_scope('biases'):
            biases = bias_variable([filter_num])  
            variable_summary(biases)
        h_conv = act(tf.nn.conv2d(input, weights, strides=strides, padding=padding) + biases)
        return h_conv


def softmax_layer(input, input_dim, output_dim, layer_name='softmax-layer'):
    """
    add a softmax layer
    """
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            variable_summary(weights)
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summary(biases)
        y = tf.nn.softmax(tf.matmul(input, weights) + biases)
        return y


def max_pooling_layer(input, k_size=[1, 2, 2, 1], strides=[1, 2, 2, 1], layer_name='max-pooling-layer', padding='SAME'):
    """
    add a max pooling layer
    """
    with tf.name_scope(layer_name):
        return tf.nn.max_pool(input, ksize=k_size, strides=strides, padding=padding)


with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')


# check
with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shaped_input, 10)  # 最多生成 10 张，注意看到的永远是最后一个 step 的中的数据对应的图片


# hidden-layer-1
dense_1 = dense_layer(x, 784, 256, layer_name='hidden-layer-1')


# dropout-layer-1
with tf.name_scope('dropout-layer-1'):
    keep_prob_1 = tf.placeholder(tf.float32)
    tf.summary.scalar('dropout_keep_probability_1', keep_prob_1)
    dropout_1 = tf.nn.dropout(dense_1, keep_prob_1)


# hidden-layer-2
dense_2 = dense_layer(dropout_1, 256, 256, layer_name='hidden-layer-2')


# dropout-layer-2
with tf.name_scope('dropout-layer-2'):
    keep_prob_2 = tf.placeholder(tf.float32)
    tf.summary.scalar('dropout_keep_probability_2', keep_prob_2)
    dropout_2 = tf.nn.dropout(dense_2, keep_prob_2)


# output layer
y = softmax_layer(dropout_2, 256, 10, layer_name='output-layer')


with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    tf.summary.scalar('cross_entropy', cross_entropy)


with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)


with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)


def feed_dict(train_flag, KEEP_PROB):
    if train_flag:
        xs, ys = mnist.train.next_batch(100)
        k = KEEP_PROB
    else:
        xs, ys = mnist.test.images, mnist.test.labels
        k = 1.0
    return {x: xs, y_: ys, keep_prob_1: k, keep_prob_2: k}


def feed_dict(train_flag, KEEP_PROB):
    if train_flag:
        xs, ys = next_batch(batch_size)
        k = KEEP_PROB
    else:
        xs, ys = X_valid, y_valid
        k = 1.0
    return {x: xs, y_: ys, keep_prob_1: k, keep_prob_2: k}


start_idx = 0
with tf.Session(config=tf.ConfigProto()) as sess:
    for KEEP_PROB in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        sess.run(tf.global_variables_initializer())
        for i in range(max_step):
            sess.run([train_step], feed_dict=feed_dict(True, KEEP_PROB))
        acc = sess.run([accuracy], feed_dict={x: X_test, y_: y_test, keep_prob_1: 1.0, keep_prob_2: 1.0})
        print('keep_prob {} accuracy {}'.format(KEEP_PROB, acc))
    # train_writer.close()
    # test_writer.close()