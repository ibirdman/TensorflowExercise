from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.externals.six import StringIO
import pydotplus
import sample.sample_set as sm
import numpy as np

SAMPLE_DATA_FILE = '../data/melon.csv'
feature_names = ['色泽',	'根蒂',	'敲声', '纹理', '脐部', '触感', '密度', '含糖率']
label_names = ['坏瓜', '好瓜']

sample_set = sm.load_samples(SAMPLE_DATA_FILE)
# print(sample_set.all.features, sample_set.all.labels)

dot_data = StringIO()
clf = tree.DecisionTreeClassifier()
train_features = np.vstack((sample_set.train.features, sample_set.validation.features))
train_labels = np.vstack((sample_set.train.labels, sample_set.validation.labels))
clf = clf.fit(train_features, train_labels)
tree.export_graphviz(clf, out_file=dot_data, feature_names=feature_names, class_names=label_names)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("melon.pdf")

print(clf.score(sample_set.test.features, sample_set.test.labels))

