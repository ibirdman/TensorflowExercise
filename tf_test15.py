import tensorflow as tf
import numpy as np

t = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -11, -12];

a = max(t)
b = t.index(max(t))
print(a)
print(b)

c = list(filter(lambda x:True if x % 3 == 0 else False, range(100)))
print(c)

l = [x for x in range(15) if x%3==0]
print(l)

c = list(filter(lambda x:True if x % 3 == 0 else False, range(100)))
print(c)

d = list(map(lambda x:x*x, t))
print('hello')
print(max(d))