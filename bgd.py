# encoding: utf-8

import pandas as pd
import numpy as np


def batchGradientDescent(x, y, theta, alpha, m, maxIterations):
    """
    x: 训练集特征，预处理中自动添加了一列全为 1 的特征（对应偏置 b）
    y: 真实值
    theta: 权重 w
    alpha: 学习率
    m: 样例个数
    maxIterations: 最大迭代次数
    """
    # 获得 x 的转置矩阵
    xTrains = x.transpose()
    # 迭代更新权值
    for i in range(0, maxIterations):
        # 计算预测值
        hypothesis = np.dot(x, theta)
        # 计算差值
        loss = hypothesis - y
        # 计算梯度
        gradient = np.dot(xTrains, loss) / m
        # 更新权值
        theta = theta - alpha * gradient
    return theta

if __name__ == '__main__':
    # 读入数据
    train_df = pd.read_csv('bgd_sgd_mbgd_train.csv')
    test_df = pd.read_csv('bgd_sgd_mbgd_test.csv')
    train_one = pd.DataFrame(np.ones(train_df.shape[0]), columns=['ONE'])
    test_one = pd.DataFrame(np.ones(test_df.shape[0]), columns=['ONE'])
    # 分离出 train_x, train_y, 并在 train_x 末尾添加一列 1
    train_x = pd.concat([train_df[['A', 'B']], train_one], axis=1, ignore_index=True)
    train_y = train_df['C']
    # 分离出 test_x, test_y, 并在 test_x 末尾添加一列 1
    test_x = pd.concat([test_df[['A', 'B']], test_one], axis=1, ignore_index=True)
    test_y = test_df['C']
    # 初始化 w, w 包含了偏置 b
    w = np.ones(train_x.shape[1])
    w = batchGradientDescent(train_x, train_y, w, 0.01, train_x.shape[0], 10000)
    print('w = %s' %w)
    print(test_y)
    print('pred: %s' %np.dot(test_x, w))

