import tensorflow as tf
import numpy as np

src = np.array([[1, 2], [3, 4]])
target = np.array([3, 4])
print(src, target)
d = list(map(lambda x: sum(np.square(x - target)), src))
print(d)
