from sklearn import svm
from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt
from sklearn.externals import joblib
import numpy as np
import sample.sample_set as sm

# 生成测试数据
out_num = 6
#X, y = make_blobs(n_samples=500, n_features=2, centers=out_num, random_state=0, cluster_std=0.8)

SAMPLE_DATA_FILE = '../data/mysamples2_circle.csv'
sample_set = sm.load_samples(SAMPLE_DATA_FILE)
X, y = sample_set.all.features, sample_set.all.labels

# 构造svm分类器实例
clf_linear = svm.SVC(C=1.0, kernel='linear')
clf_poly = svm.SVC(C=1.0, kernel='poly', degree=3)
clf_rbf = svm.SVC(C=1.0, kernel='rbf', gamma=0.5)
clf_rbf2 = svm.SVC(C=1.0, kernel='rbf', gamma=0.1)

plt.figure(figsize=(10, 10), dpi=144)

clfs = [clf_linear, clf_poly, clf_rbf, clf_rbf2]
titles = ['Linear Kernel',
            'Polynomial Kernel with Degree=3',
            'Gaussian Kernel with gamma=0.5',
            'Gaussian Kernel with gamma=0.1']

# train and predict
for clf, i in zip(clfs, range(len(clfs))):
    clf.fit(X, y)
    print("{}'s score:{}".format(titles[i], clf.score(X,y)))
    out = clf.predict(X)
    print("out's shape:{}, out:{}".format(out.shape, out))
    # plt.subplot(2, 2, i+1)
    # plot_hyperplane(clf, X, y,  title=titles[i])

# 参考页面：http://scikit-learn.org/stable/modules/model_persistence.html
# http://sofasofa.io/forum_main_post.php?postid=1001002
# save trained model to disk-file
for clf, i in zip(clfs, range(len(clfs))):
    joblib.dump(clf, str(i)+'.pkl')

# load model from file and test
for i in range(len(clfs)):
    clf = joblib.load(str(i)+'.pkl')
    print("{}'s score:{}".format(titles[i], clf.score(X, y)))


OUTPUT_CLASS_STYLES = np.array([
['0', 'o', 'r'],
['1', 'x', 'g'],
['2', 'v', 'b'],
['3', 's', 'k'],
['4', '*', 'y'],
['5', '^', '#808000'],
])

def draw_train_data(X, Y):
    # Y = np.argmax(Y, 1);
    for i in range(out_num):
        index = np.where(Y==i)[0]
        label = OUTPUT_CLASS_STYLES[i, 0]
        marker = OUTPUT_CLASS_STYLES[i, 1]
        color = OUTPUT_CLASS_STYLES[i, 2]
        plt.scatter(X[index,0], X[index,1], marker = marker, color = color, label = label, s = 30)

def draw_pred_map(X, Y):
    # Y = np.argmax(Y, 1);
    for i in range(out_num):
        index = np.where(Y == i)[0]
        label = OUTPUT_CLASS_STYLES[i, 0]
        marker = OUTPUT_CLASS_STYLES[i, 1]
        color = OUTPUT_CLASS_STYLES[i, 2]
        plt.scatter(X[index, 0], X[index, 1], marker=marker, color=color, label=label, s=15, alpha=0.1)


def create_predict_map(X):
    x1_num = 100
    x2_num = 100
    x_map = np.zeros(shape=(x1_num * x2_num, 2), dtype=np.float)
    X1 = np.linspace(min(X[:, 0]), max(X[:, 0]), num=x1_num)
    X2 = np.linspace(min(X[:, 1]), max(X[:, 1]), num=x2_num)
    num = 0
    for x1 in X1:
        for x2 in X2:
            x_map[num, 0] = x1
            x_map[num, 1] = x2
            num += 1
    return x_map

x_pred_map = create_predict_map(X)
# print(x_pred_map)
y_pred_map = clf_rbf.predict(x_pred_map)
print(y_pred_map)
# print(np.argmax(y_pred_map, 1))
draw_pred_map(x_pred_map, y_pred_map)
draw_train_data(X, y)

plt.xlabel('X1')
plt.ylabel('X2')
plt.title('sample points')
plt.show()