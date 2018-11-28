import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

c1_x = [[1, 1], [2, 2], [3,3], [4,4]]
c1_x = np.transpose(c1_x)
plt.scatter(c1_x[0], c1_x[1], c = 'b', marker = 'x', edgecolor='none', s=30)

c2_x = [[1, 3], [2, 5], [3,7], [4,9]]
c2_x = np.transpose(c2_x)
plt.scatter(c2_x[0], c2_x[1], c = 'r', marker = 'o', edgecolor='none', s=40)
plt.show()
