import tensorflow as tf
import numpy as np

c = np.array([[[1.,2], [3.,4], [5.,6]], [[3.,5], [5.,6], [6.,7]]])
print(c[:,:,0].shape)

d = np.array([1, 2, 3])
print(d.std())

e = 2./3
print(np.sqrt(e))

img_noise = np.random.randint(1, 9, size=(2, 5))
print(img_noise)