#!/usr/bin/env python
# -*- encoding:utf-8 -*-

"""
@Author  :   Q.W.Wang
@Software:   PyCharm
@File    :   sigmod.py
@Time    :   2018-08-25 15:23
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import math

x = np.arange(-10, 10, 0.01)
y = np.array([1 / (1 + math.exp(-1 * i)) for i in x])
fig = plt.figure()
plt.plot(x, y, ls='-', lw=1, c='r')
plt.grid(True)
plt.show()
