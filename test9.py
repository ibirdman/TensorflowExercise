from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np

a = np.array([[0.],[1.]])
b = np.where(a == 0)
print(b)
print(a[b])

