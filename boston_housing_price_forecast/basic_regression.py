# encoding: utf-8
"""
Python 的每个新版本都会增加一些新的功能，或者对原来的功能作一些改动。
有些改动是不兼容旧版本的，也就是在当前版本运行正常的代码，到下一个版本运行就可能不正常了。
Python提供了 __future__ 模块，把下一个新版本的特性导入到当前版本，于是我们就可以在当前版本中测试一些新版本的特性。
"""
from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow import keras
import numpy as np
print(tf.__version__)
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 在服务器端使用 matplotlib
import matplotlib.pyplot as plt


"""
定义参数
"""
EPOCHS = 500


"""
下载数据集
"""
boston_housing = keras.datasets.boston_housing
(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()


"""
check
"""
print('train_data.shape:', train_data.shape)
print('len(train_labels):', len(train_labels))
print('test_data.shape:', test_data.shape)
print('len(test_labels):', len(test_labels))


"""
乱序
"""
order = np.argsort(np.random.random(train_labels.shape))
train_data = train_data[order]
train_labels = train_labels[order]
print("Training set: {}".format(train_data.shape))  # 404 examples, 13 features
print("Testing set:  {}".format(test_data.shape))   # 102 examples, 13 features
print('train_data[0]:', train_data[0])


"""
转成 DataFrame
"""
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
                'TAX', 'PTRATIO', 'B', 'LSTAT']
df = pd.DataFrame(train_data, columns=column_names)


"""
标准化
"""
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std
print(train_data[0])


"""
建模
"""
def build_model():
    model = keras.Sequential([
        keras.layers.Dense(64, activation=tf.nn.relu,
                        input_shape=(train_data.shape[1],)),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(1)
    ])
    optimizer = tf.train.RMSPropOptimizer(0.001)
    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])  # 评价指标是 MAE
    return model
model = build_model()
model.summary()


"""
自定义回调函数
"""
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print('')
        print('.', end='')


"""
早停，如果 20 个epoch 都没有让 val_loss 减小，那么就提前结束训练
"""
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)


"""
训练
"""
history = model.fit(train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    callbacks=[early_stop, PrintDot()])

print('')
print(history.epoch)
print(np.array(history.history['mean_absolute_error']))
print(np.array(history.history['val_mean_absolute_error']))


"""
画图
"""
def plot_history(history, pIdx):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [1000$]')
    plt.plot(history.epoch, np.array(history.history['mean_absolute_error']), label='Train Loss')
    plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']), label = 'Val loss')
    plt.legend()
    plt.ylim([0, 5])
    plt.savefig('basic_regression_{}.png'.format(pIdx))
plot_history(history, 1)


"""
模型评估
"""
[loss, mae] = model.evaluate(test_data, test_labels, verbose=0)
"""
7.2f: 占位符宽度为7（包括两位小数点和符号“.”，不足 7 位则在开头填空格，多于 7 位则保留 7 位即可），保留小数点后两位（四舍五入）
"""
print('Mean Abs Error: ${}'.format(mae))
print("Testing set Mean Abs Error: ${:7.2f}".format(mae))
