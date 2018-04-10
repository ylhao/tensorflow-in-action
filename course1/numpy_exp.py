# encoding: utf-8

import math
import numpy as np

v = np.random.rand(10, 1)
u = np.zeros((10, 1))

for i in range(10):
    u[i] = math.exp(v[i])
print(u)

u = np.exp(v)
print(u)
