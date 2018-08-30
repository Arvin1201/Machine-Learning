#!/usr/bin/env python
# -*- encoding:utf-8 -*-

'''
@Author  :   Q.W.Wang
@Software:   PyCharm
@File    :   0_加载线性回归的模型.py
@Time    :   2018-08-19 22:10
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.externals import joblib  # 保存模型时使用,只能作用于python中
import matplotlib as mpl
from matplotlib import  pyplot as plt
import time

# 中文显示设置
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False

pd.set_option('display.width', 1000)
np.set_printoptions(threshold=1000)
pd.set_option('display.width', 1000)  # 设置字符显示宽度
pd.set_option('display.max_rows', 50)  # 设置显示最大行
pd.set_option('display.max_columns', None)  # 设置显示最大列，None为显示所有列
# 1、加载数据
#  数据存放的路径
path = r'../../../datas/0_regression/household_power_consumption_1000.txt'
df = pd.read_csv(path, sep=';')  # 读取数据


# # 异常数据处理(异常数据过滤)
datas = df.dropna(how='any',axis=0)
datas = datas.replace('?',np.nan)

# 预测时间与功率之间的关系
def getDateTime(x):
    """
    进行时间分割成年月日 时分秒
    :param x:
    :return:
    """
    t = time.strptime(' '.join(x),'%d/%m/%Y %H:%M:%S')   # 将序列中的元素连接成一个字符串
    return (t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec)

# 取出特征点
datetime = datas.iloc[:,0:2]
X = datetime.apply(lambda x:pd.Series(getDateTime(x)),axis=1)  # 获取X的值
X.columns=['Year','Month','Day','Hour','Minis','Second']
Y = datas.iloc[:,2:3]  # 获取Y值

# 训练集与测试集的划分
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size= 0.2,random_state=28)

ss = joblib.load( "../../../model/0_regression/DateTime_Power_ss.model")  # 加载归一化模型
lr = joblib.load("../../../model/0_regression/DateTime_Power_data_lr.model")  # 加载训练好的模型
print(lr.coef_)
print(lr.intercept_)

X_test = ss.transform(X_test)
Y_predict = lr.predict(X_test)

t=np.arange(len(X_test))
plt.figure(facecolor='w')#建一个画布，facecolor是背景色
plt.plot(t, Y_test, 'r-', linewidth=2, label='真实值')
plt.plot(t, Y_predict, 'g-', linewidth=2, label='预测值')
plt.legend(loc = 'upper left') #显示图例，设置图例的位置
plt.title("线性回归预测时间和功率之间的关系", fontsize=20)
plt.grid(b=True)  #加网格
plt.show()

