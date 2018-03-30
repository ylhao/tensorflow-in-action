# encoding: utf-8

import numpy as np

# 训练数据
train_set = np.array([[3,3,1],[4,3,1],[1,1,-1]])
train_y = train_set[:, -1]
train_x = train_set[:, 0:2]
# 计算 Gram Matrix
Gram = np.dot(train_x, train_x.T)
print('Gram Matrix = %s' %Gram)

# 初始化 w, b
w = [0,0]
b = 0
# 初始化 α
a = np.zeros(len(train_set), np.float)

def cal(i):
    global a, b, train_x, train_y
    res = np.dot(a*train_y, Gram[i])
    res = (res + b) * train_y[i]
    return res

def update(i):
    global a, b
    a[i] += 1
    b += train_y[i]
    print('α = %s, b = %s' %(a, b))

def check():
    global a, b, train_x, train_y
    flag = False
    for i in range(train_set.shape[0]):
        # 检查是否还有分错的点
        if cal(i) <= 0:
            flag = True
            update(i)
    if not flag:
        w = np.dot(a*train_y, train_x)
        print('w = %s, b = %s' %(w, b))
        return False
    return True;

if __name__ == '__main__':
    flag = False
    for i in range(1000):
        if not check():  # 判断是否全部都划分正确了
            flag = True
            break
    if flag:
        print('经过 %s 次迭代全部正确' %(i + 1))
    else:
        print('经过 1000 次迭代，仍有点划分错误')
