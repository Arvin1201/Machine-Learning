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
    b = np.random.normal(-1, 1, size)
    c = np.random.standard_normal(size)
    return b, c

def primitiveFunc(x, b, c):
    sum1 = 0
    for i in range(len(b)):
        y1 = x ** 2 + b[i] * x + c[i]
        sum1 += y1 / len(b)
    return sum1


def derivedFunc(x, b):
    return x * 2 + b


def gradientDescentCalc_SGD(x, b, c,alpha,count,tol):
    """
    随机梯度下降法计算
    :param x: 初始值
    :param b: 特征值b
    :param c:  特征值c
    :param alpha:学习率
    :param count:迭代次数
    :param tol:迭代的误差
    :return: 返回X与Y
    """
    X = []
    Y = []
    Change = []
    current_y = primitiveFunc(x,b, c)
    step_change = 1.0 + tol
    Y.append(current_y)
    Change.append(step_change)
    change_numbers = 0  # 变量修改的次数
    # 迭代计算
    # 迭代次数
    step = 0
    while step_change > tol and step < count:
        """
        SGD是每条样本更新
        在随机梯度中，是使用每条样本更新一次模型参数(这里的模型参数就是x)；
        这里总共有：n条样本，那也就是说在一次epoch中，更新参数n次。
        """

        # 打乱随机顺序
        random_index = np.random.permutation(len(b))
        for index1 in random_index:
            # 计算函数的梯度值
            SGDalue = derivedFunc(x, b[index1])
            # 基于SGD更新模型参数
            x = x - alpha * SGDalue
            # 更新x所对应的y值
            tmp_y = primitiveFunc(x,b,c)
            # 记录一下变化大小
            step_change = np.abs(tmp_y - current_y)
            Change.append(step_change)
            current_y = tmp_y
            Y.append(current_y)
            change_numbers += 1
            # 如果step_change的值达到要求了进行推出
            if step_change < tol:
                break
        step += 1
    return Y,Change,step,change_numbers


def viewPlot(Y,step):
    fig = plt.figure()
    sub1 = plt.subplot(1, 2, 1)
    sub1.plot(range(step + 1), Y, ls='-', c='r', lw=2)
    sub1.set_xlabel('迭代次数')
    sub1.set_ylabel('每次迭代的y值变化大小')
    sub1.set_title("SGD")
    sub1.grid(True)
    plt.show()


if __name__ == '__main__':
    numcount = 1000
    count = 1000
    np.random.seed(28)
    b = np.random.normal(-1, 1, numcount)
    c = np.random.standard_normal(numcount)
    # 10 --> 0.05~0.09    100 -> 0.005~0.009   1000 -> 0.0005~0.0009   0.9/len(b)
    Y, Change, step,change_numbers = gradientDescentCalc_SGD(np.random.randint(low=-10, high=10), b, c,0.01,count,0.00001)  # 10 --> 0.05~0.9
    print('SGD迭代次数{}'.format(step))
    viewPlot(Y,change_numbers)
