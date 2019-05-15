from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.externals.six import StringIO
import pydotplus
import numpy as np

feature_color_name = np.array(['浅白', '青绿', '乌黑'])
bak = feature_color_name[:]
print(feature_color_name, bak)
#feature_color_name.append('abc')
np.random.shuffle(feature_color_name)
print(feature_color_name, bak)
