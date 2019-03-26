import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

data = []
label = []
np.random.seed(0) # 设置随机数生成时所用算法开始的整数值

# 以原点为圆心，半径为1的圆把散点划分成红蓝两部分，并加入随机噪音。
for i in range(150):
    x1 = np.random.uniform(-1,1)  # 随机生成下一个实数，它在 [-1，1) 范围内。
    x2 = np.random.uniform(0,2)
    if x1**2 + x2**2 <= 1:
        data.append([np.random.normal(x1, 0.1),np.random.normal(x2,0.1)])
        label.append(0)
    else:
        data.append([np.random.normal(x1, 0.1), np.random.normal(x2, 0.1)])
        label.append(1)

data = np.hstack(data).reshape(-1,2) # 把数据转换成n行2列
label = np.hstack(label).reshape(-1, 1)  # 把数据转换为n行1列
plt.scatter(data[:,0], data[:,1], c=np.squeeze(label), cmap="RdBu", vmin=-.2, vmax=1.2, edgecolor="white")
plt.show()
