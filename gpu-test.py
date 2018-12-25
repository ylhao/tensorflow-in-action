# encoding: utf-8

import tensorflow as tf

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
    a = tf.constant(1)
    b = tf.constant(3)
    c = a + b
    print('c={}'.format(sess.run(c)))

# allow_soft_placement=True 表示不能使用 GPU 时选择使用 CPU，因为不是所有的操作都可以被放在 GPU 上，如果强行将无法放在 GPU 上的操作指定到 GPU 上，将会报错。
# log_device_placement=True 可以查看你的操作和张量被分配到了哪个设备上进行计算
