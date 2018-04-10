# encoding: utf-8

import numpy as np

# a 的 shape 为 (5,)，既不是行向量也不是列向量
# a 只是秩为 1 的数组
a = np.random.randn(5)
print(a.shape)

# 定义一个列向量
a = np.random.randn(5, 1)
print(np.dot(a, a.T))
print(a.shape)
a = np.random.randn(5).reshape(5, 1)
print(a.shape)
a = np.array([[1], [2], [3]])
print(a.shape)

# 定义一个行向量
a = np.random.randn(1, 5)
print(a.shape)
a = np.random.randn(5).reshape(1, 5)
print(a.shape)
a = np.array([[1, 2, 3]])
print(a.shape)

assert(a.shape == (1, 3))
print('success (1, 3)')
# 会产生 AssertionError
# assert(a.shape == (2, 3))
# print('success (2, 3)')
