#!/usr/bin/env python
# -*- encoding:utf-8 -*-

"""
@Author  :   Q.W.Wang
@Software:   PyCharm
@File    :   随机森林.py
@Time    :   2018-08-24 21:26
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import tree
# from sklearn.ensemble

pd.set_option('display.width', 1000)
np.set_printoptions(threshold=1000)
pd.set_option('display.width', 1000)  # 设置字符显示宽度
pd.set_option('display.max_rows', 50)  # 设置显示最大行
pd.set_option('display.max_columns', None)  # 设置显示最大列，None为显示所有列