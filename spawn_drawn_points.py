import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 使用 NumPy 生成假数据(phony data), 总共 20 个点.
x_data = np.float32(np.random.rand(1, 20)) # 随机输入
y_data = np.dot([2], x_data) + 5
# print(x_data)
# print(y_data)

fix,ax = plt.subplots(figsize = (5, 5))
ax.scatter(x_data,y_data,s = 30,c = 'b',marker = 'o',label = 'point')
ax.legend()
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
plt.show()
