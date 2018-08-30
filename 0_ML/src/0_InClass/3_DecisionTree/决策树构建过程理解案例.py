#!/usr/bin/env python
# -*- encoding:utf-8 -*-

"""
@Author  :   Q.W.Wang
@Software:   PyCharm
@File    :   决策树构建过程理解案例.py
@Time    :   2018-08-26 15:22
"""

import numpy as np
import math


# 信息熵
def entropy(t):
    return np.sum([-i * np.log2(i) for i in t])



