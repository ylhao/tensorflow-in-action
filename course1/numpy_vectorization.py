# encoding: utf-8

import time
import numpy as np

a = np.random.rand(1000000)
b = np.random.rand(1000000)

tic = time.time()
c = np.dot(a, b)
toc = time.time()
print(c)
print('Vectorization version: %s ms' %(toc - tic))

c = 0
tic = time.time()
for i in range(1000000):
    c += a[i] * b[i]
toc = time.time()
print(c)
print('For loop version: %s ms' %(toc - tic))

'''
numpy.random.rand: https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.rand.html
numpy.dot: https://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html
通常用 np.dot 来做向量运算（内积和）
'''
