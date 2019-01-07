import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

def draw_predict_map():
    X1 = np.arange(0, 10, 0.5)
    X2 = np.arange(0, 10, 0.5)
    print(tf.Session().run(X))
    for x1 in X1:
        for x2 in X2:
           
           # plt.scatter(x1, x2, c='green', edgecolors='none', s=10)
           plt.scatter(x1,x2,cmap=['green', 'red'],s=20)

# plt.scatter(x,y,c=y,cmap=plt.cm.gist_rainbow,s=20)

print(np.array([1, 2], dtype=np.float))

fig = plt.figure()
draw_predict_map()
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend(loc = 'upper left')
plt.show()
