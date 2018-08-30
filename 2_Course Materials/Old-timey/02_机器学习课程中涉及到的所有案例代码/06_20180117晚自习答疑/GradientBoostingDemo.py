# --encoding:utf-8 --
"""根据梯度下降的公式求解线性回归"""

import math
import numpy as np


def validate(X, Y):
    """校验X和Y的数据格式是否正确"""
    if len(X) != len(Y):
        raise Exception("参数异常")
    else:
        m = len(X[0])
        for l in X:
            if len(l) != m:
                raise Exception("参数异常")
        if len(Y[0]) != 1:
            raise Exception("参数异常")


def calcGD(x, y, theta, xj):
    """计算梯度值"""
    # 计算ax - y的值
    lx = len(x)
    la = len(theta)
    if lx == la:
        result = 0
        for i in range(lx):
            result += x[i] * theta[i]
        return (y - result) * xj
    elif lx + 1 == la:
        result = 0
        for i in range(lx):
            result += x[i] * theta[i]
        result += 1 * theta[lx]  # 加上常数项
        return (y - result) * xj
    else:
        raise Exception("参数异常")


def fit(X, Y, alpha=0.01, max_iter=100, add_constant_item=True, threshold=10e-4):
    """
    进行模型训练, 使用BGD的思想进行构造
    X: 输入的特征矩阵，要求是一个二维的数组
    Y: 输入的是目标矩阵，要求是一个二维的数组
    alpha: 学习率
    max_iter: 最大迭代次数
    """
    # 1. 校验数据格式是否正确
    validate(X, Y)

    # 2. 开始构建模型
    sample_numbers = len(X)
    features_numbers = len(X[0])
    # 当需要添加常数项的时候，认为特征多一个
    features_numbers = features_numbers + 1 if add_constant_item else features_numbers
    max_iter = 100 if max_iter <= 0 else max_iter
    theta = [0 for i in range(features_numbers)]
    for i in range(max_iter):
        # 临时保存所有变量的导数值
        for j in range(features_numbers):
            # 迭代更新每个theta值（theta_j）
            # a. 获取所有的梯度和
            ts = 0.0
            for k in range(sample_numbers):
                # 将sample_numbers个样本的导数值进行相加
                if j == features_numbers - 1 and add_constant_item:
                    gd = calcGD(X[k], Y[k][0], theta, 1)
                else:
                    gd = calcGD(X[k], Y[k][0], theta, X[k][j])
                ts += gd

            # b. 更新theta值
            theta[j] += alpha * ts

        # 所有的theta值更新一遍了，那么判断算法是否收敛
        sum_j = 0.0
        for k in range(sample_numbers):
            sum_j += math.pow(calcGD(X[k], Y[k][0], theta, 1), 2)
        sum_j /= sample_numbers

        # 判断误差是否小于某个值
        if sum_j < threshold:
            break

    # 返回最终的参数
    return theta


# 预测结果
def predict(X, a):
    Y = []
    n = len(X[0])
    has_constant_item = len(a) > n
    for x in X:
        result = 0
        for i in range(n):
            result += x[i] * a[i]

        # 添加常数项
        if has_constant_item:
            result += a[n]

        # 添加结果
        Y.append(result)
    return Y


# 计算实际值和预测值之间的相关性
def calcRScore(y, py):
    if len(y) != len(py):
        raise Exception("参数异常")
    import math
    import numpy as np
    avgy = np.average(y)
    m = len(y)
    rss = 0.0
    tss = 0
    for i in range(m):
        rss += math.pow(y[i] - py[i], 2)
        tss += math.pow(y[i] - avgy, 2)
    r = 1.0 - 1.0 * rss / tss
    return r


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import sklearn
from sklearn.linear_model import LinearRegression, Ridge, LassoCV, RidgeCV, ElasticNetCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model.coordinate_descent import ConvergenceWarning

## 设置字符集，防止中文乱码
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

# warnings.filterwarnings(action = 'ignore', category=ConvergenceWarning)
## 创建模拟数据
np.random.seed(0)
np.set_printoptions(linewidth=1000, suppress=True)
N = 10
x = np.linspace(0, 6, N) + np.random.randn(N)
y = 1.8 * x ** 3 + x ** 2 - 14 * x - 7 + np.random.randn(N)
x.shape = -1, 1
y.shape = -1, 1
print(x)
print(y)

plt.figure(figsize=(12, 6), facecolor='w')

## 模拟数据产生
x_hat = np.linspace(x.min(), x.max(), num=100)
x_hat.shape = -1, 1

## 线性模型
model = LinearRegression()
model.fit(x, y)
y_hat = model.predict(x_hat)
s1 = calcRScore(y, model.predict(x))
print(model.score(x, y))  ## 自带R^2输出
print("模块自带实现===============")
print("参数列表:", model.coef_)
print("截距:", model.intercept_)

## 自模型
ma = fit(x, y, alpha=0.01, max_iter=100, add_constant_item=True)
y_hat2 = predict(x_hat, ma)
s2 = calcRScore(y, predict(x, ma))
print("自定义实现模型=============")
print("参数列表:", ma)

## 开始画图
plt.plot(x, y, 'ro', ms=10, zorder=3)
plt.plot(x_hat, y_hat, color='#b624db', lw=2, alpha=0.75, label=u'Python模型，$R^2$:%.3f' % s1, zorder=2)
plt.plot(x_hat, y_hat2, color='#6d49b6', lw=2, alpha=0.75, label=u'自己实现模型，$R^2$:%.3f' % s2, zorder=1)
plt.legend(loc='upper left')
plt.grid(True)
plt.xlabel('X', fontsize=16)
plt.ylabel('Y', fontsize=16)

plt.suptitle(u'自定义的线性模型和模块中的线性模型比较', fontsize=22)
plt.show()
