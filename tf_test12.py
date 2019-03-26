import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

xx, yy = np.mgrid[0:10:1, 0:10:1]

grid = np.c_[xx.ravel(), yy.ravel()]
print(grid)

def create_predict_map():
    x1_num = 10
    x2_num = 10
    # x_map = np.zeros(shape=(x1_num * x2_num, 2), dtype=np.float)
    X1 = np.linspace(0, 10, num=x1_num)
    X2 = np.linspace(0, 10, num=x2_num)
    x_map = np.c_[X1.ravel(), X2.ravel()]
    print(x_map)
    return x_map

grid = create_predict_map()
plt.scatter(grid[:,0], grid[:,1], vmin=-10, vmax=10, edgecolor="white", s=15)
plt.show()
