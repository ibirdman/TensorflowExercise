import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

class logistic(object):
    def __init__(self):
        self.W = None
    def train(self,X,y,learn_rate = 0.01,num_iters = 5000):
        num_train,num_feature = X.shape
        #init the weight
        self.W = 0.001*np.random.randn(num_feature,1).reshape((-1,1))
        loss = []
        
        for i in range(num_iters):
            error,dW = self.compute_loss(X,y)
            self.W += -learn_rate*dW
            
            loss.append(error)
            if i%200==0:
                print('i=%d,error=%f' %(i,error))
        return loss
    
    def compute_loss(self,X,y):
        num_train = X.shape[0]
        h = self.output(X)
        loss = -np.sum((y*np.log(h) + (1-y)*np.log((1-h))))
        loss = loss / num_train
        
        dW = X.T.dot((h-y)) / num_train
    
        return loss,dW
    
    def output(self,X):
        g = np.dot(X,self.W)
        return self.sigmod(g)
    def sigmod(self,X):
        return 1/(1+np.exp(-X))
    
    def predict(self,X_test):
        h = self.output(X_test)
        y_pred = np.where(h>=0.5,1,0)
        return y_pred


c1_x = [[1, 1], [2, 2], [3,3], [4,4], [3, 2], [2.5, 2.5]]
c1_y = [1 for i in range(len(c1_x))]
print(c1_x)
c2_x = [[1, 3], [2, 3], [3,4], [4,5], [3, 3.5], [2.5, 4]]
c2_y = [0 for i in range(len(c2_x))]
print(c2_x)

X = np.vstack((c1_x, c2_x))
y = np.hstack((c1_y, c2_y))
print(X)
print(y)
 
y = y.reshape((-1,1))
#add the x0=1
one = np.ones((X.shape[0],1))
X_train = np.hstack((one,X))
classify = logistic()
loss = classify.train(X_train,y)
print(classify.W)
 
# draw
fig = plt.figure()
ax = fig.add_subplot(111)

label = np.array(y)
index_0 = np.where(label==0)
plt.scatter(X[index_0,0],X[index_0,1],marker='x',color = 'b',label = '0',s = 15)
index_1 =np.where(label==1)
plt.scatter(X[index_1,0],X[index_1,1],marker='o',color = 'r',label = '1',s = 15)
 
#show the decision boundary
x1 = np.arange(0,10,0.5)
x2 = (- classify.W[0] - classify.W[1]*x1) / classify.W[2]

plt.plot(x1,x2,color = 'black')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend(loc = 'upper left')

def on_press(event):
    print("my position:" ,event.button,event.xdata, event.ydata)
    plt.scatter(event.xdata,event.ydata,marker='o',color = 'r',label = '1',s = 15)
    plt.show()
    
fig.canvas.mpl_connect('button_press_event', on_press)

plt.show()

