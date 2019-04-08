from functools import reduce
import numpy as np

a=[1,2,4,2,4,5,6,5,7,8,9,0]

b={}

b=b.fromkeys(a)
print(type(b))

c=list(b.keys())

print(c)