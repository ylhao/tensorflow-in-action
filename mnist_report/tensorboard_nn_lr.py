# encoding: utf-8


import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import shutil
from sklearn.utils import shuffle


batch_size = 100
learning_rate = 0.001
KEEP_PROB = 0.9
max_step = 2500
decay_step = 100
decay_rate = 0.96
data_dir = './tmp/data/mnist/input_data'  # 数据
log_dir = './nn_lr_tmp/logs/mnist_with_summaries'  # 日志
ckpt_dir = './nn_lr_tmp/ckpt'  # 模型


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


def dense_layer(input, input_dim, output_dim, layer_name='dense-layer', act=tf.nn.relu):
    """
    add a dense layer
    """
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):  # 该层权重
            weights = weight_variable([input_dim, output_dim])
        with tf.name_scope('biases'):  # 该层偏置
            biases = bias_variable([output_dim])
        with tf.name_scope('Wx_plus_b'):  # 线性变换
            preactivate = tf.matmul(input, weights) + biases
        activations = act(preactivate, name='activation')  # 非线性变换
        return activations  # 返回非线性变换的结果


def softmax_layer(input, input_dim, output_dim, layer_name='softmax-layer'):
    """
    add a softmax layer
    """
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
        y = tf.nn.softmax(tf.matmul(input, weights) + biases)
        return y


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


# cross entropy
with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    tf.summary.scalar('cross_entropy', cross_entropy)


# train
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)


# accuracy
with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)


def feed_dict(train_flag):
    if train_flag:
        xs, ys = mnist.train.next_batch(100)
        k = KEEP_PROB
    else:
        xs, ys = X_valid, y_valid
        k = 1.0
    return {x: xs, y_: ys, keep_prob_1: k, keep_prob_2: k}


clean_dir(log_dir)  # 清空文件夹
merged = tf.summary.merge_all()  # 将之前定义的所有summary op 整合到一起，获取所有的之前定义的汇总操作
with tf.Session(config=tf.ConfigProto()) as sess:
    train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(log_dir + '/test')
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    for i in range(1, max_step + 1):
        if i % 10 == 0:
            summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
            train_writer.add_summary(summary, i)
            summary, = sess.run([merged], feed_dict=feed_dict(False))
            test_writer.add_summary(summary, i)
        else:
            _, = sess.run([train_step], feed_dict=feed_dict(True))
        if i % decay_step == 0 and i != 0:
            learning_rate = learning_rate * decay_rate
        if i % 500 == 0 and i != 0:
            acc = sess.run([accuracy], feed_dict=feed_dict(False))
            print('Accuracy at step %s: %s' %(i, acc))
            save_path = saver.save(sess, ckpt_dir + '/model.ckpt', global_step=i)
            print('Model saved in file: %s' %save_path)
    acc = sess.run([accuracy], feed_dict={x: X_test, y_: y_test, keep_prob_1: 1.0, keep_prob_2: 1.0})
    print('accuracy: %s' %(acc))
    train_writer.close()
    test_writer.close()
