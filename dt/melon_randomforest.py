from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
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
clf = RandomForestClassifier(max_depth=4)
#拟合模型
clf.fit(train_features, train_labels)

Estimators = clf.estimators_
for index, model in enumerate(Estimators):
    filename = 'melon_' + str(index) + '.pdf'
    dot_data = tree.export_graphviz(model , out_file=None,
                         feature_names=feature_names,
                         class_names=label_names,
                         filled=True, rounded=True,
                         special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    # 使用ipython的终端jupyter notebook显示。
    Image(graph.create_png())
    graph.write_pdf(filename)

print(clf.score(sample_set.test.features, sample_set.test.labels))
