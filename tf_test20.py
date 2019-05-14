from functools import reduce
import numpy as np

a=np.array([1,2,4,2,4,5,7,10,5,5,7,8,9,0,3])

b = np.arange(1, 5).reshape(2, 2)
print(b)

c = np.arange(1, 5).reshape(2, 2)
print(c)

d = np.dot(b, c)
print(d)