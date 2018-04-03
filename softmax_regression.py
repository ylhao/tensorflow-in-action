# encoding: utf-8

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)

# 会把 sess 这个 session 注册为默认的 session
sess = tf.InteractiveSession()

# 创建一个 placeholder，placeholder 就是输入数据的地方
# 参数是输入输入数据的类型和 shape
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

# Variable 不同于存储数据的 tensor，Variable 是持久化存储的，tensor 一旦用掉就会消失
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 计算预测值 y
y = tf.nn.softmax(tf.matmul(x, W) + b)

# 计算交叉熵
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# 定义进行训练的操作
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# 定义一个全局参数初始化器并直接执行其 run() 方法，对参数进行初始化
tf.global_variables_initializer().run()

for i in range(0, 1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))

