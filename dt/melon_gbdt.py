from sklearn import datasets
from sklearn.ensemble import GradientBoostingClassifier
from IPython.display import Image
from sklearn import tree
import pydotplus
import sample.sample_set as sm
import numpy as np
from sklearn.metrics import accuracy_score

SAMPLE_DATA_FILE = '../data/melon.csv'
feature_names = ['色泽',	'根蒂',	'敲声', '纹理', '脐部', '触感', '密度', '含糖率']
label_names = ['坏瓜', '好瓜']

sample_set = sm.load_samples(SAMPLE_DATA_FILE)
train_features = np.vstack((sample_set.train.features, sample_set.validation.features))
train_labels = np.vstack((sample_set.train.labels, sample_set.validation.labels))

# 训练模型，限制树的最大深度4
clf = GradientBoostingClassifier()
#拟合模型
clf.fit(train_features, train_labels)

print(clf.score(sample_set.test.features, sample_set.test.labels))
