# encoding: utf-8

import numpy as np

# 下面几行代码帮助理解，我们可以认为 numpy 生成的是我们说的列向量
x = np.array([[1,1,1],[2,2,2],[3,3,3]])
print(x[:, -1].shape)
print(np.ones(3).shape)
print(np.ones((3,)).shape)
# 手动定义了一个列向量，三行一列
print(np.array([[1],[1],[1]]).shape)
# 定义行向量的正确形式如下，一行三列
print(np.array([[1,1,1]]).shape)

"""
numpy 元素级别的运算
对应位置的运算
区别于矩阵运算
"""
a = np.ones(3)
b = np.ones(3)
print(a*b)
print(np.dot(a, b.T))
