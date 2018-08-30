#!/usr/bin/env python
# -*- encoding:utf-8 -*-

"""
@Author  :   Q.W.Wang
@Software:   PyCharm
@File    :   0_线性回归.py
@Time    :   2018-08-19 16:22
"""

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
# 进行数据归一化
ss=StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)

# 选择线性回归模型并进行模型训练
lr =LinearRegression()
lr.fit(X_train,Y_train)
Y_predict = lr.predict(X_test)

# 保存模型
joblib.dump(ss, "../../../model/0_regression/DateTime_Power_ss.model") ## 将标准化模型保存
joblib.dump(lr, "../../../model/0_regression/DateTime_Power_data_lr.model") ## 将模型保存

# print("训练R2:",lr.score(X_train, Y_train))
# print("测试R2:",lr.score(X_test, Y_test))
# mse = np.average((Y_predict-Y_test)**2)
# rmse = np.sqrt(mse)
# print("rmse:",rmse)
# print(lr.coef_)
# print(lr.intercept_)

# 画图
"""
t=np.arange(len(X_test))
plt.figure(facecolor='w')#建一个画布，facecolor是背景色
plt.plot(t, Y_test, 'r-', linewidth=2, label='真实值')
plt.plot(t, Y_predict, 'g-', linewidth=2, label='预测值')
plt.legend(loc = 'upper left') #显示图例，设置图例的位置
plt.title("线性回归预测时间和功率之间的关系", fontsize=20)
plt.grid(b=True)#加网格
plt.show()
"""

# 功率与电流之间的关系
X2= datas.iloc[:,2:4]
Y2= datas.loc[:,'Global_intensity']

# 训练集与测试集的划分
X2_train,X2_test,Y2_train,Y2_test = train_test_split(X2,Y2,test_size=0.8,random_state=28)

# 数据归一化处理
ss2 = StandardScaler()
X2_train = ss2.fit_transform(X2_train)
X2_test = ss2.transform(X2_test)

# 模型选择
lr2 = LinearRegression()
lr2.fit(X2_train,Y2_train)
print(lr2.intercept_)

joblib.dump(ss2, r'../../../model/0_regression/Power_Current_ss.model')
joblib.dump(lr2, r'../../../model/0_regression/Power_Current_lr.model')

Y2_predict = lr2.predict(X2_test)

# 输出衡量指标
print(u'训练R2:{0}'.format(lr2.score(X2_train,Y2_train)))
print(u'测试R2:{0}'.format(lr2.score(X2_test,Y2_test)))
print('RMSE:{0}'.format(np.sqrt(np.average((Y2_predict - Y2_test)**2))))

# 画图进行比较
fig = plt.figure(facecolor='w')
c=np.arange(len(X2_test))
plt.plot(c,Y2_test,'r-',lw=2)
plt.plot(c,Y2_predict,'g-',lw=1)
plt.grid(True)
plt.show()







