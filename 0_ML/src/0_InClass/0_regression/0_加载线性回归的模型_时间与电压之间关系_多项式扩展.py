#!/usr/bin/env python
# -*- encoding:utf-8 -*-

"""
@Author  :   Q.W.Wang
@Software:   PyCharm
@File    :   0_加载线性回归的模型_时间与电压之间关系_多项式扩展.py
@Time    :   2018-08-29 21:00
"""

"""
    多项式扩展进行预测时间与功率之间的关系
"""

# 1、导包
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.externals import joblib
import time

# 2、设置显示(包括中文显示，pycharm中的数据显示)
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', None)

# 3、加载数据
path = r'../../../datas/0_regression/household_power_consumption_1000.txt'
df = pd.read_csv(path, sep=';')
# print(df.head(5))

# 3、异常数据处理
new_df = df.replace('?',np.nan)
datas = new_df.dropna(axis=0,how='any')

# 4、获取特征与标签
def getDataTime(x):
    '''
    获取时间，返回的是pd集合
    :param x:
    :return:
    '''
    strtime = time.strptime(' '.join(x), '%d/%m/%Y %H:%M:%S')  # '%d/%m/%Y %H:%M:%S'
    return pd.Series([strtime.tm_year, strtime.tm_mon, strtime.tm_mday, strtime.tm_hour, strtime.tm_min, strtime.tm_sec])

X_DataTime = (datas.loc[:,'Date':'Time']).apply(lambda x:getDataTime(x),axis = 1)
Y_Voltage = datas.loc[:,'Voltage']

# 5、对数据集进行划分为测试集与训练集
X_DataTime_train,X_DataTime_test,Y_Voltage_train,Y_Voltage_test = train_test_split(X_DataTime,Y_Voltage,test_size=0.2,random_state=28)

# 6、数据的标准化
ss = StandardScaler()
## 进行数据的标准化，其中fit_transform中的Y参数是没有用的，查看源码就知道
X_DataTime_train = ss.fit_transform(X_DataTime_train,Y_Voltage_train)
X_DataTime_test = ss.transform(X_DataTime_test)
joblib.dump(ss,"../../../model/0_regression/时间与电压_多项式_标准化.model")

# 7、模型训练
## 采用管道流的形式进行训练：
##      将多个操作合并成为一个操作
##      Pipleline总可以给定多个不同的操作，给定每个不同操作的名称即可，执行的时候，按照从前到后的顺序执行
##      Pipleline对象在执行的过程中，当调用某个方法的时候，会调用对应过程的对应对象的对应方法
##      eg：在下面这个案例中，调用了fit方法
##          那么对数据调用第一步操作：PolynomialFeatures的fit_transform方法对数据进行转换并构建模型
##          然后对转换之后的数据调用第二步操作: LinearRegression的fit方法构建模型
##      eg: 在下面这个案例中，调用了predict方法
##      那么对数据调用第一步操作：PolynomialFeatures的transform方法对数据进行转换
##      然后对转换之后的数据调用第二步操作: LinearRegression的predict方法进行预测
##
models = [
    Pipeline(
        [
            ('Poly',PolynomialFeatures()),
            ('Linaer',LinearRegression())
        ]
    )
]
model = models[0]
## 根据不同的多项式阶数进行扩展，从而训练模型
X_Size = len(X_DataTime_test)
t_X = np.arange(X_Size)
N = 5
degree_pool = np.arange(1,N,1)
color = []
## 产生5中颜色
for c in np.linspace(int('0xffff00',16),255,N):
    color.append('#%06x' % int(c))

colors = ['y','k','g','c']
## 绘制一个窗口，并设置窗口的大小与颜色
fig1 = plt.figure(figsize = (20,10 ),facecolor='w')
for i,d in enumerate(degree_pool):
    sub = plt.subplot(N - 1,1,i+1)
    # plt.subplot(4,1,i+1)
    sub.plot(t_X,Y_Voltage_test,ls='-',lw=1,c='r')
    ## 进行管道中模型训练
    model.set_params(Poly__degree=d)   # 1阶
    model.fit(X_DataTime_train,Y_Voltage_train)
    ## 获取模型对象
    lr = model.get_params('Linaer')['Linaer']
    output = u'%d阶,系数为:' % d
    joblib.dump(model,"../../../model/0_regression/时间与电压_多项式_{0}阶.model".format(d))
    Y_hat = model.predict(X_DataTime_test)
    score = model.score(X_DataTime_test,Y_Voltage_test)
    print(score)
    print(color[i])
    sub.plot(t_X,Y_hat,ls='-',lw=2,c=color[i],label = u'%d阶,准确率为:%0.3f' % (d,score))
    sub.legend(loc = 'upper left')
    sub.grid(True)
    sub.set_ylabel(u'%d阶结果' % d)
plt.show()



