# encoding: utf-8

# 训练数据
train_set = [[(3,3),1],[(4,3),1],[(1,1),-1]]

# 初始化 w, b
w = [0,0]
b = 0

def judge(item):
    # 计算 wx + b
    res = 0;
    for i in range(len(item[0])):
        res += item[0][i] * w[i]
    res += b
    # 计算 (wx + b) * y
    res *= item[1]
    return res

def check():
    flag = False
    for item in train_set:
        if judge(item) <= 0:  # 如果还有误分类点
            flag = True
            update(item)
    return flag

def update(item):
    global w, b
    # 更新 w0: w0 + αxiyi, α = 1
    w[0] += 1 * item[1] * item[0][0]
    # 更新 w1
    w[1] += 1 * item[1] * item[0][1]
    # 更新偏置 b
    b += 1 * item[1]
    print('w = %s, b = %s' %(w, b))

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
