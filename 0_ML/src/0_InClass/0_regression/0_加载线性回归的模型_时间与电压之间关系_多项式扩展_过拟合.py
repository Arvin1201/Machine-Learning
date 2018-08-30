#!/usr/bin/env python
# -*- encoding:utf-8 -*-

"""
@Author  :   Q.W.Wang
@Software:   PyCharm
@File    :   0_加载线性回归的模型_时间与电压之间关系_多项式扩展_过拟合.py
@Time    :   2018-08-29 23:00
"""
"""
不是多项式的介绍越高越好，越高可能就会出现多拟合
"""
 # 1 导包
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures

# 2 解决中文乱码及pycharm中的数据显示问题
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False
pd.set_option('display.width',1000)
pd.set_option('display.max_rows',50)
pd.set_option('display.max_columns',50)

# 3 加载数据
path = r'../../../datas/0_regression/household_power_consumption_1000.txt'
df = pd.read_csv(path,sep=';')

# 异常数据处理
new_df = df.replace('?',np.nan)
datas = new_df.dropna(axis = 'index',how='any')














