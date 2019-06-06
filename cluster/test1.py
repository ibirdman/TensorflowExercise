#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.metrics import homogeneity_completeness_v_measure
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import fcluster


class ClusterMethod:

    def __init__(self):
        l1=np.zeros(100)
        l2=np.ones(100)
        self.labels=np.concatenate((l1,l2),)

    #随机创建两个二维正太分布，形成数据集
    def dataProduction(self):
        # 随机创建两个二维正太分布，形成数据集
        np.random.seed(4711)
        c1 = np.random.multivariate_normal([10, 0], [[3, 1], [1, 4]], size=[100, ])
        l1 = np.zeros(100)
        l2 = np.ones(100)
        # 一个100行的服从正态分布的二维数组
        c2 = np.random.multivariate_normal([0, 10], [[3, 1], [1, 4]], size=[100, ])
        # 加上一些噪音
        np.random.seed(1)
        noise1x = np.random.normal(0, 2, 100)
        noise1y = np.random.normal(0, 8, 100)
        noise2 = np.random.normal(0, 8, 100)
        c1[:, 0] += noise1x  # 第0列加入噪音数据
        c1[:, 1] += noise1y
        c2[:, 1] += noise2

        # 定义绘图
        self.fig = plt.figure(figsize=(20, 15))
        # 添加子图，返回Axes实例，参数：子图总行数，子图总列数，子图位置
        ax = self.fig.add_subplot(111)
        # x轴
        ax.set_xlabel('x', fontsize=30)
        # y轴
        ax.set_ylabel('y', fontsize=30)
        # 标题
        self.fig.suptitle('classes', fontsize=30)
        # 连接
        labels = np.concatenate((l1, l2), )
        X = np.concatenate((c1, c2), )
        # 散点图
        pp1 = ax.scatter(c1[:, 0], c1[:, 1], cmap='prism', s=50, color='r')
        pp2 = ax.scatter(c2[:, 0], c2[:, 1], cmap='prism', s=50, color='g')
        ax.legend((pp1, pp2), ('class 1', 'class 2'), fontsize=35)
        self.fig.savefig('scatter.png')
        return X

    def clusterMethods(self):
        X=self.dataProduction()
        self.fig.clf()#reset plt
        self.fig,((axis1,axis2),(axis3,axis4))=plt.subplots(2,2,sharex='col',sharey='row')#函数返回一个figure图像和一个子图ax的array列表

        #k-means
        self.kMeans(X,axis1)
        #mean-shift
        self.DBScan(X,axis2)
        #gaussianMix
        self.gaussianMix(X,axis3)
        #hierarchicalWard
        self.hierarchicalWard(X,axis4)

    def kMeans(self,X,axis1):
        kmeans=KMeans(n_clusters=2)#聚类个数
        kmeans.fit(X)#训练
        pred_kmeans=kmeans.labels_#每个样本所属的类
        print('kmeans:',np.unique(kmeans.labels_))
        print('kmeans:',homogeneity_completeness_v_measure(self.labels,pred_kmeans))#评估方法，同质性，完整性，两者的调和平均
        #plt.scatter(X[:,0],X[:,1],c=kmeans.labels_,cmap='prism')
        axis1.scatter(X[:,0],X[:,1],c=kmeans.labels_,cmap='prism')
        axis1.set_ylabel('y',fontsize=40)
        axis1.set_title('k-means',fontsize=40)
        #plt.show()

    def DBScan(self, X, axis1):
        dbscan = DBSCAN(eps = 0.1)
        pred_dbs = dbscan.fit_predict(X)  # 训练
        print('dbscan:', np.unique(pred_dbs))
        axis1.scatter(X[:, 0], X[:, 1], c=pred_dbs, cmap='prism')
        axis1.set_ylabel('y', fontsize=40)
        axis1.set_title('dbscan', fontsize=40)
        # plt.show()

    def meanShift(self,X,axis2):
        ms=MeanShift(bandwidth=7)#带宽
        ms.fit(X)
        pred_ms=ms.labels_
        axis2.scatter(X[:,0],X[:,1],c=pred_ms,cmap='prism')
        axis2.set_title('mean-shift',fontsize=40)
        print('mean-shift:',np.unique(ms.labels_))
        print('mean-shift:',homogeneity_completeness_v_measure(self.labels,pred_ms))

    def gaussianMix(self,X,axis3):
        gmm=GaussianMixture(n_components=2)
        gmm.fit(X)
        pred_gmm=gmm.predict(X)
        axis3.scatter(X[:, 0], X[:, 1], c=pred_gmm, cmap='prism')
        axis3.set_xlabel('x', fontsize=40)
        axis3.set_ylabel('y', fontsize=40)
        axis3.set_title('gaussian mixture', fontsize=40)
        print('gmm:',np.unique(pred_gmm))
        print('gmm:',homogeneity_completeness_v_measure(self.labels,pred_gmm))

    def hierarchicalWard(self,X,axis4):
        ward=linkage(X,'ward')#训练
        max_d=110#终止层次算法最大的连接距离
        pred_h=fcluster(ward,max_d,criterion='distance')#预测属于哪个类
        axis4.scatter(X[:,0], X[:,1], c=pred_h, cmap='prism')
        axis4.set_xlabel('x',fontsize=40)
        axis4.set_title('hierarchical ward',fontsize=40)
        print('ward:',np.unique(pred_h))
        print('ward:',homogeneity_completeness_v_measure(self.labels,pred_h))

        self.fig.set_size_inches(18.5,10.5)
        self.fig.savefig('comp_clustering.png',dpi=100)#保存图

if __name__=='__main__':
    cluster=ClusterMethod()
    cluster.clusterMethods()