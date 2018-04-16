# encoding: utf-8

import numpy as np


def basic_sigmoid(x):
     """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or numpy array of any size

    Return:
    s -- sigmoid(x)
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
	 """
    Compute the gradient (also called the slope or derivative) of the sigmoid function with respect to its input x.
    You can store the output of the sigmoid function into variables and then use it to calculate the gradient.

    Arguments:
    x -- A scalar or numpy array

    Return:
    ds -- Your computed gradient.
    """
	s = sigmoid(x)
	ds = s * (1 - s)
	return ds


def image2vector(image):
	"""
    Argument:
    image -- a numpy array of shape (length, height, depth)

    Returns:
    v -- a vector of shape (length*height*depth, 1)
    """
	return image.reshape(image.shape[0] * image.shape[1] * image.shape[2], 1)


def normalizeRows(x):
	"""
    Implement a function that normalizes each row of the matrix x (to have unit length).

    Argument:
    x -- A numpy matrix of shape (n, m)

    Returns:
    x -- The normalized (by row) numpy matrix. You are allowed to modify x.
    """
	# 注意这里的 keepdims
	"""
	linalg = linear + algbra = 线性代数
	norm: 求范数
	axis: 列方向
    keepdim: 是否将范数计算结果和原来矩阵维度对应，如果设为 False，那么一个两行散列的矩阵对应的计算结果的维度为 (2,)，如果设为 True，那么对应的维度为 (2, 1)
	"""
	x_norm = np.linalg.norm(x, axis = 1, keepdims = True)
	x = x / x_norm
	return x


def softmax(x):
	 """Calculates the softmax for each row of the input x.

    Your code should work for a row vector and also for matrices of shape (n, m).

    Argument:
    x -- A numpy matrix of shape (n,m)

    Returns:
    s -- A numpy matrix equal to the softmax of x, of shape (n,m)
    """
	x_exp = np.exp(x)
	x_sum = np.sum(x_exp, axis = 1, keepdims = True)
	s = x_exp / x_sum
	return s


def L1(yhat, y):
	"""
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)

    Returns:
    loss -- the value of the L1 loss function defined above
    """
    loss = np.sum(np.abs(yhat - y))
    return loss


def L2(yhat, y):
	"""
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)

    Returns:
    loss -- the value of the L2 loss function defined above
    """
	# loss = np.sum((yhat - y) ** 2)
	loss = np.dot((yhat - y), (yhat - y).T)
	return loss


















