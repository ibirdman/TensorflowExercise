from sklearn import datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import sample.sample_set as sm

SAMPLE_DATA_FILE = '../data/melon.csv'
feature_names = ['色泽',	'根蒂',	'敲声', '纹理', '脐部', '触感', '密度', '含糖率']
label_names = ['坏瓜', '好瓜']

sample_set = sm.load_samples(SAMPLE_DATA_FILE)

### data analysis
print(sample_set.all.features.shape)   # 输入空间维度
print(sample_set.all.labels.shape) # 输出空间维度

### data split
x_train,x_test,y_train,y_test = train_test_split(sample_set.all.features,
                                                  sample_set.all.labels,
                                                  test_size = 0.3,
                                                  random_state = 33)

### fit model for train data
model = XGBClassifier()
model.fit(x_train, y_train)

Estimators = model.estimators_
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


print(model.score(sample_set.test.features, sample_set.test.labels))
