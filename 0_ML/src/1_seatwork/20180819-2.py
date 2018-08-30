#!/usr/bin/env python
# -*- encoding:utf-8 -*-

"""
@Author  :   Q.W.Wang
@Software:   PyCharm
@File    :   20180819-2.py
@Time    :   2018-08-21 21:01
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 中文显示设置
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

"""
梯度下降作业：
  目标函数：
    y = x**2 + b * x + c
  需求：求解最小值对应的x和y(这里的最小值指的是函数y在所有数据上的一个最小值)
  要去：写代码
    数据：
		b: 服从均值为-1，方差为10的随机数
		c：服从均值为0，方差为1的随机数
	假定b、c这样的数据组合总共1、2、10、100、10w、100w条数据,求解在现在的数据
	情况下，目标函数的取最小值的时候，x和y分别对应多少？
"""


def bAndcValues(size):
    """
    获取b和c的随机值
    :param size: 获取随机值的大小
    :return:
    """
    b = np.random.normal(-1, 10, size)
    c = np.random.standard_normal(size)
    return b, c


def getMin(x, b, c):
    return np.mean(np.array(b))


def primitiveFunc(x, b, c):
    sum1 = 0
    for i in range(len(b)):
        y1 = x ** 2 + b[i] * x + c[i]
        sum1 += y1 / len(b)
    return sum1


def derivedFunc(x, b):
    sum2 = 0
    for i in range(len(b)):
        y2 = x * 2 + b[i]
        sum2 += y2/ len(b)
    return sum2


def gradientDescentCalc(x, b, c, step, count):
    """
    梯度下降法计算
    :param x: 初始值
    :param b: 特征值b
    :param c:  特征值c
    :param step:
    :param count:迭代次数
    :return: 返回X与Y
    """
    X = []
    Y = []
    f_diff = primitiveFunc(x, b, c)
    f_cur = f_diff
    X.append(x)
    Y.append(f_cur)
    # 迭代计算
    index = 0
    while f_diff > 1e-10 and index < count:
        index += 1
        x = x - step * derivedFunc(x, b)
        temp = primitiveFunc(x, b, c)
        f_diff = np.fabs(f_cur - temp)
        f_cur = temp
        X.append(x)
        Y.append(temp)
    return X, Y, index


def viewPlot(x, y, b, c, index):
    fig = plt.figure()
    X = np.arange(-5.0, 5.0, 0.01)
    Y = primitiveFunc(X, b, c)
    plt.grid(True)
    print('b == ', b)
    print('c == ', c)
    sub1 = plt.subplot(2, 2, 1)
    sub1.hist(b, bins=1000)
    sub2 = plt.subplot(2, 2, 2)
    sub2.hist(c, bins=1000)
    sub3 = plt.subplot(212)
    sub3.plot(X, Y, ls='-', c='r', lw=2)
    sub3.plot(x, y, ls='--', c='g', lw=1.5)
    b_values = getMin(x[-1], b, c)
    plt.title(u'样本容量为{0}下的最终解为:x={1},ymin={2}\n迭代次数{3},所求的b={4}'.format(len(b), x[-1], y[-1], index, b_values))
    plt.show()


if __name__ == '__main__':
    numcount = 1000
    count = 1000000
    np.random.seed(28)
    b = np.random.normal(-1, 10.0, numcount)
    c = np.random.standard_normal(numcount)
    # 10 --> 0.05~0.09    100 -> 0.005~0.009   1000 -> 0.0005~0.0009   0.9/len(b)
    X, Y, index = gradientDescentCalc(np.random.randint(low=-10, high=10), b, c, 0.001, count)  # 10 --> 0.05~0.9
    viewPlot(X, Y, b, c, index)
