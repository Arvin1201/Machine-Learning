#!/usr/bin/env python
# -*- encoding:utf-8 -*-

'''
@Author  :   Q.W.Wang
@Software:   PyCharm
@File    :   20180819作业.py
@Time    :   2018-08-20 21:27
'''

import scipy.optimize as opt
import numpy as np

# from sympy import *
# x = symbols("x")  # 符号x，自变量
# y = -1*(0.5*x**3)*(0.3+0.4*x)**5*(0.7-0.9*x)**2
#
# print(diff(y,x))

def func1(x):
    return -1*(0.5*x**3)*(0.3+0.4*x)**5*(0.7-0.9*x)**2

def h(x):
    return -1.0*x**3*(-0.9*x + 0.7)**2*(0.4*x + 0.3)**4 - 0.5*x**3*(0.4*x + 0.3)**5*(1.62*x - 1.26) - 1.5*x**2*(-0.9*x + 0.7)**2*(0.4*x + 0.3)**5

x=10
f_change=func1(x)
f_current=func1(x)
X_N = []
Y_N = []
X_N.append(f_change)
X_N.append(f_current)
step = 0.1
count = 0
while f_change > 1e-10 and count < 200:
    x = x - step*h(x)
    f_change = np.fabs(f_change - func1(x))
    f_current = func1(x)
    X_N.append(x)
    Y_N.append(f_current)
print(X_N)
print(Y_N)
