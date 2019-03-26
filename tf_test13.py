import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def height(x,y):
    return (1-x/2+x**5+y**3)*np.exp(-x**2-y**2)

def create_predict_map(X):
    num = 10
    xmin = min(X[:, 0])
    xmax = max(X[:, 0])
    xunit = (xmax - xmin) / num
    print(xmin, xmax, xunit)
    ymin = min(X[:, 1])
    ymax = max(X[:, 1])
    yunit = (ymax - ymin) / num
    print(ymin, ymax, yunit)

    xx, yy = np.mgrid[xmin:xmax:xunit, ymin:ymax:yunit]
    xy_map = np.c_[xx.ravel(), yy.ravel()]
    return xy_map

num = 5
xmin = 0
xmax = 4
xunit = (xmax - xmin) / num

ymin = 0
ymax = 4
yunit = (ymax - ymin) / num

xx, yy = np.mgrid[xmin:xmax:xunit, ymin:ymax:yunit]
xy_map = np.c_[xx.ravel(), yy.ravel()]

print(xx)
# print(xy_map)
print(xy_map[:,0], xy_map[:,1])
print(height(xx, yy).reshape(xy_map[:,0].shape))
print(height(xx, yy))

# plt.contourf(xx, yy, height(xx, yy), 8)
plt.contourf(xx, yy, height(xx, yy), 8)
plt.show()