from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np

a = [[1,2], [3,4], [5,6]]
b = [[5,6], [7,8], [9,0]]
c = np.vstack((a, b))
print(c.T)
