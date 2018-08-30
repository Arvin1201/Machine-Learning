#!/usr/bin/env python
# -*- encoding:utf-8 -*-

"""
@Author  :   Q.W.Wang
@Software:   PyCharm
@File    :   鸢尾花数据分类.py
@Time    :   2018-08-22 18:39
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier  # 分类数
from sklearn import tree
from IPython.display import Image
import pydotplus

pd.set_option('display.width',1000)
pd.set_option('display.max_rows',100)
pd.set_option('display.max_columns',100)

# 加载数据
path = r'../../datas/3_DecisionTree/iris.data'
# 定义常量
isirs_name = ['sepal length','seqal width','petal length','petal width','cla']
# 2 查看数据的特性
df = pd.read_csv(path,sep=',',low_memory=True,header=None,names=isirs_name)
# 2 查看数据的特性
# print(df.info())
# print(df.describe().T)
# print(df.head(5))
# 3 从原始数据中获取x，y
X = df.iloc[:,0:4]
Y = df.loc[:,'cla']
Y = pd.Categorical(Y).codes   # 将类别转换成数字
# print(Y)
# 4 数据的划分
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=28)
print("训练集样本数量:%d" % X_train.shape[0])
print("测试集样本数量:%d" % X_test.shape[0])

# 5 模型的训练  采用决策树相关模型进行分类
## scikit-learn中决策树默认都是CART模型
"""
class sklearn.tree.DecisionTreeClassifier(criterion=’gini’, 
splitter=’best’, max_depth=None, min_samples_split=2,
 min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None,
  random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0,
   min_impurity_split=None, class_weight=None, presort=False)
criterion=’gini’  --> 给定树构建过程中需要考虑的指标，默认为Gini系数，也就是CART
可选entropy作为指标
splitter=’best’--> 给定选择划分属性的时候，采用什么方式，默认为best；表示选择最优的方式
max_depth=None  --> 树的最大深度，默认不限制
min_samples_split  --> 当结点中的样本数量小于等于该值的时候，停止树的构建
min_samples_leaf   -->  最小叶子结点的数量
max_features   --> 防止过拟合
"""
tree1 = DecisionTreeClassifier(criterion='gini',max_depth=10)
tree1.fit(X_train,Y_train)  # 模型训练
print(tree1.predict_proba(X_train))

dot_data = tree.export_graphviz(tree1, out_file=None,
                         feature_names=['sepal length', 'sepal width', 'petal length', 'petal width'],
                         class_names=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],
                         filled=True, rounded=True,
                         special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf("iris2.pdf")
#Image(graph.create_png())

# 模型相关指标输出
print("训练集上的准确率：%.2f" % tree1.score(X_train,Y_train))
print("测试集上的准确率：%.2f" % tree1.score(X_test,Y_test))
y_hat = tree1.predict(X_test)
print(np.mean(y_hat == Y_test))
print("每个样本所属的类的预测概率信息")
# print(tree.predict_proba(X_test))
# 在决策数构建的时候，将影响信息增益更大的属性是放在树的上面的节点中进行判断；
# 也就是说，可以认为决策树构建的树中，越往上的节点，作用越强 ==> 所以可以基于决策树做
# 特征选择，实际代码中就是feature_importances_参数输出中最大的k个指标
# print("各个特征的重要性指标：",end="")
# print(tree.feature_importances_)  # 查看各个特征中影响的重要新














