# encoding: utf-8


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import shutil
import os


max_step = 1000
learning_rate = 0.001
dropout = 0.9
data_dir = './tmp/data/mnist/input_data'  # 数据
log_dir = './tmp/logs/mnist_with_summaries'  # 日志
ckpt_dir = './tmp/ckpt'  # 模型


mnist = input_data.read_data_sets(data_dir, one_hot=True)
# xs, ys = mnist.train.next_batch(100)
# print(xs[0])
# print(ys[0])


def safe_mkdir(path):
    """
    如果目录不存在，创建目录
    :param path: 目录名
    """
    try:
        os.mkdir(path)
    except OSError:
        pass


def clean_dir(path):
    """
    清空文件夹
    :param path: 目录名
    """
    shutil.rmtree(path)
    safe_mkdir(path)


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def variable_summary(var):
    with tf.name_scope('summaries'):
        with tf.name_scope('mean'):
            mean = tf.reduce_mean(var)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        with tf.name_scope('max'):
            max = tf.reduce_max(var)
        with tf.name_scope('min'):
            min = tf.reduce_min(var)
        tf.summary.scalar('mean', mean)  # 均值
        tf.summary.scalar('stddev', stddev)  # 标准差
        tf.summary.scalar('max', max)  # 最大值
        tf.summary.scalar('min', min)  # 最小值
        tf.summary.histogram('histogram', var)  # 使用 tf.summary.histogram 记录变量 var 的直方图


def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """
    :param input_tensor: 输入的 tensor
    :param input_dim: 输入 tensor 的维度
    :param output_dim: 输出 tensor 的维度
    :param layer_name: 该层的名称（方便在 tensorboard 中查看网络结构）
    :param act: 激活函数，默认使用 relu
    :return: 返回计算结果
    """
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            variable_summary(weights)
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summary(biases)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram('preactivations', preactivate)
        activations = act(preactivate, name='activation')
        tf.summary.histogram('activation', activations)
        return activations


with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    #  利用tf.summary.image 将图片信息汇总展示
    tf.summary.image('input', image_shaped_input, 10)  # 最多生成 10 张，注意看到的永远是最后一个 step 的中的数据对应的图片

# hidden layer
hidden1 = nn_layer(x, 784, 500, 'hidden-layer')

# dropout
with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    tf.summary.scalar('dropout_keep_probability', keep_prob)
    dropped = tf.nn.dropout(hidden1, keep_prob)

# output layer
y = nn_layer(dropped, 500, 10, 'output-layer')

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
    return {x: xs, y_: ys, keep_prob: k}


# 清空文件夹
clean_dir(log_dir)
# 将之前定义的所有summary op 整合到一起，获取所有的之前定义的汇总操作
merged = tf.summary.merge_all()
with tf.Session() as sess:
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

