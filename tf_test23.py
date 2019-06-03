from sklearn.datasets.samples_generator import make_blobs
from matplotlib import pyplot as plt

X,y=make_blobs(n_samples=100,n_features=2,centers=5,random_state=0,cluster_std=0.6) #n_samples=50意思取50个点，centers=2意思是将数据分为两
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap='autumn')
plt.show()
