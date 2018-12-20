# encoding: utf-8


import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import shutil



max_step = 1000
learning_rate = 0.001
dropout = 0.8
data_dir = './tmp/data/mnist/input_data'  # 数据
log_dir = './tmp/logs/mnist_with_summaries'  # 日志
ckpt_dir = './tmp/ckpt'  # 模型


mnist = input_data.read_data_sets(data_dir, one_hot=True)
# mnist = tf.keras.datasets.mnist
# (x_train, y_train),(x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0
# xs, ys = mnist.train.next_batch(100)
# print(xs[0])
# print(ys[0])


def safe_mkdir(path):
    """
    如果目录不存在，创建目录
    """
    if not os.path.exists(path):
        os.mkdir(path)


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


def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """
    :param input_tensor: 输入的 tensor
    :param input_dim: 输入 tensor 的维度
    :param output_dim: 输出 tensor 的维度
    :param layer_name: 该层的名称
    :param act: 激活函数，默认使用 relu
    :return: 返回计算结果
    """
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):  # 该层权重
            weights = weight_variable([input_dim, output_dim])
            variable_summary(weights)
        with tf.name_scope('biases'):  # 该层偏置
            biases = bias_variable([output_dim])
            variable_summary(biases)
        with tf.name_scope('Wx_plus_b'):  # 线性变换
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram('preactivations', preactivate)  # 线性变换结果的直方图
        activations = act(preactivate, name='activation')  # 非线性变换
        tf.summary.histogram('activation', activations)  # 非线性变换结果的直方图
        return activations  # 返回非线性变换的结果


with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')


with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shaped_input, 10)  # 最多生成 10 张，注意看到的永远是最后一个 step 的中的数据对应的图片

# hidden-layer-1
hidden_layer_1 = nn_layer(x, 784, 256, 'hidden-layer-1')

# dropout-layer-1
with tf.name_scope('dropout-1'):
    keep_prob_1 = tf.placeholder(tf.float32)
    tf.summary.scalar('dropout_keep_probability_1', keep_prob_1)
    dropped_1 = tf.nn.dropout(hidden_layer_1, keep_prob_1)

# hidden-layer-2
hidden_layer_2 = nn_layer(dropped_1, 256, 256, 'hidden-layer-2')

# dropout-layer-2
with tf.name_scope('dropout-2'):
    keep_prob_2 = tf.placeholder(tf.float32)
    tf.summary.scalar('dropout_keep_probability_2', keep_prob_2)
    dropped_2 = tf.nn.dropout(hidden_layer_2, keep_prob_2)

# output layer
y = nn_layer(dropped_2, 256, 10, 'output-layer')

with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
    tf.summary.scalar('cross_entropy', cross_entropy)

with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    # y: 100 * 10, y_: 100 * 10
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, axis=1), tf.argmax(y_, axis=1)), tf.float32))
    tf.summary.scalar('accuracy', accuracy)


def feed_dict(train):
    if train:
        xs, ys = mnist.train.next_batch(100)
        k = dropout
    else:
        xs, ys = mnist.test.images, mnist.test.labels
        k = 1.0
    return {x: xs, y_: ys, keep_prob_1: k, keep_prob_2: k}


clean_dir(log_dir)  # 清空文件夹
merged = tf.summary.merge_all()  # 将之前定义的所有summary op 整合到一起，获取所有的之前定义的汇总操作
with tf.Session(config=tf.ConfigProto()) as sess:
    train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(log_dir + '/test')
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    for i in range(max_step):
        if i % 10 == 0:
            summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
            test_writer.add_summary(summary, i)
            print('Accuracy at step %s: %s' %(i, acc))
        else:
            if i % 100 == 99:
                # 下面两行可以记录训练运行时间和内存占用等方面的信息
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata= tf.RunMetadata()
                summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True),
                                      options=run_options, run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, 'step%03d' %i)
                train_writer.add_summary(summary, i)
                save_path = saver.save(sess, ckpt_dir + '/model.ckpt', global_step=i)
                print('Model saved in file: %s' %save_path)
                print('Adding run metadata for', i)
            else:
                summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
                train_writer.add_summary(summary, i)
    train_writer.close()
    test_writer.close()
