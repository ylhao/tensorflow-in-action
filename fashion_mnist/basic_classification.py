# encoding: utf-8

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 服务器端使用 matplotlib
import matplotlib.pyplot as plt
print('tensorflow version:', tf.__version__)  # 打印 tensorflow 的版本信息


# 下载数据集
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


# 定义类别名称
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# check
print('train_images.shape:', train_images.shape)
print('len(train_labels):', len(train_labels))
print('test_images.shape:', test_images.shape)
print('len(test_labels):', len(test_labels))


# 查看一张图片
plt.figure()
"""
matplotlib.pyplot.imshow(X, cmap=None)  # cmap: 颜色图谱（colormap), 默认绘制为RGB(A)颜色空间。
"""
plt.imshow(train_images[0])
plt.colorbar()
"""
plt.grid(True)  显示背景的网格线
plt.grid(False)  关闭背景的网格线
ax.grid(color=’r’, linestyle=’-‘, linewidth=2)  设置背景网格线的样式
"""
plt.grid(False)
plt.savefig("train_images_0.png")


# 缩放
train_images = train_images / 255.0
test_images = test_images / 255.0


# 显示 25 张图片
plt.figure(figsize=(10,10))
for i in range(25):
    """
    subplot(nrows, ncols, index, **kwargs)
    """
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.savefig("train_images_0_24.png")


# 定义模型
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])


# 编译模型
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# 训练模型
model.fit(train_images, train_labels, epochs=10)


# 评估模型
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test accuracy:', test_acc)


# 预测
predictions = model.predict(test_images)
print('predictions[0]:', predictions[0])
print('np.argmax(predictions[0]):', np.argmax(predictions[0]))
print('test_labels[0]:', test_labels[0])


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel("{} {:2.0f}% (true label: {})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    xlocation =  np.linspace(1, len(class_names) * 2, len(class_names))
    # print(xlocation)
    plt.xticks(xlocation, class_names, fontsize=6, rotation=30)
    plt.yticks([])
    """
    matplotlib.pyplot.ylim(*args, **kwargs)  设置坐标轴范围
    """
    plt.ylim((0, 1))
    thisplot = plt.bar(xlocation, predictions_array, width=2, color="#777777")
    predicted_label = np.argmax(predictions_array)
    """
    预测结果对应标签的颜色为红色
    真实标签对应的颜色为蓝色
    """
    thisplot[true_label].set_color('blue')
    thisplot[predicted_label].set_color('red')
    # 添加数据标签
    for plot, predict_prob in zip(thisplot, predictions_array):
        height = plot.get_height()
        predict_prob = int(predict_prob * 10000) / 100
        if predict_prob == 0:
            continue
        else:
            predict_prob = '{}%'.format(predict_prob)
            plt.text(plot.get_x(), height, predict_prob, fontsize=6, va='bottom')


i = 0
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.savefig("predictions_0.png")


num_rows = 5
num_cols = 3
num_images = num_rows*num_cols  # 图片数量
plt.figure(figsize=(8*num_cols, 4*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.savefig("predictions_0_24.png")


"""
预测单张图片
"""
img = test_images[0]
print('img.shape:', img.shape)
"""
把图片添加到一个 batch 中（这个 batch 仅包含这一张图片）
"""
img = (np.expand_dims(img,0))
print('img.shape:', img.shape)
predictions_single = model.predict(img)
print('predictions_single:', predictions_single)
print('np.argmax(predictions_single[0]):', np.argmax(predictions_single[0]))
