# encoding: utf-8

import numpy as np

def sigmod(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

# 输入
X = np.array([[0.35],[0.9]])
# 真实值
y = np.array([0.5])

W0 = np.array([[0.1, 0.8], [0.4, 0.6]])
# W1 = np.array([0.3, 0.9])  # 如果这样定义，定义的是一个列向量，shape 为 (2,)
W1 = np.array([[0.3, 0.9]])

print('original W0: %s' %W0)
print('original W1: %s' %W1)

for j in range(100):
    l0 = X
    # 隐藏层的输出
    l1 = sigmod(np.dot(W0, l0))
    # 输出层的输出（预测值）
    l2 = sigmod(np.dot(W1, l1))
    # 反向传播
    l2_error = y - l2
    error = (y - l2)**2 / 2.0
    print('error: %s' %error)
    l2_delta = l2_error * sigmod(l2, deriv=True)
    l1_error = l2_delta * W1
    l1_delta = l1_error * sigmod(l1, deriv=True)
    #修改权值
    W1 += l2_delta * l1.T
    W0 += l0.T.dot(l1_delta)

print('W0: %s' %W0)
print('W1: %s' %W1)
