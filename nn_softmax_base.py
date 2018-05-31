# encoding: utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)

# 会把 sess 这个 session 注册为默认的 session
sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.truncated_normal([784, 300], stddev=0.1))
b1 = tf.Variable(tf.zeros([300]))
W2 = tf.Variable(tf.zeros([300, 10]))
b2 = tf.Variable(tf.zeros([10]))

hidden = tf.nn.relu(tf.matmul(x, W1) + b1)
hidden_drop = tf.nn.dropout(hidden, keep_prob)
y = tf.nn.softmax(tf.matmul(hidden_drop, W2) + b2)

# 计算交叉熵
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.AdagradOptimizer(0.3).minimize(cross_entropy)
tf.global_variables_initializer().run()
for i in range(0, 3000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys, keep_prob: 0.75})
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
