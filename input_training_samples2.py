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
 #           if i%200==0:
 #               print('i=%d,error=%f' %(i,error))
 
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

def on_press(event):
    global samples_data
    print("my position:" ,event.button,event.xdata, event.ydata)
    if event.button == 1: cla = 1
    elif event.button == 3: cla = 0
    samples_data.append([event.xdata,event.ydata,cla])  
    # print(samples_data)
    X, y, W = start_train(samples_data)
    plt.clf()
    redraw(X, y, W)
    
def start_train(sample_data):
    samples = np.array(samples_data)
    X = samples[:,0:2]
    y = samples[:,2]
    # print(X)
    # print(y)    
     
    y = y.reshape((-1,1))
    #add the x0=1
    one = np.ones((X.shape[0],1))
    X_train = np.hstack((one,X))
    classify = logistic()
    loss = classify.train(X_train,y)
    print(classify.W)
    return X, y, classify.W

def redraw(X, y, W):
    label = np.array(y)
    index_0 = np.where(label==0)
    plt.scatter(X[index_0,0],X[index_0,1],marker='x',color = 'b',label = '0',s = 15)
    index_1 =np.where(label==1)
    plt.scatter(X[index_1,0],X[index_1,1],marker='o',color = 'r',label = '1',s = 15)
     
    #show the decision boundary
    x1 = np.arange(0,10,0.5)
    x2 = (- W[0] - W[1]*x1) / W[2]

    plt.plot(x1,x2,color = 'black')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend(loc = 'upper left')
    plt.show()
    
samples_data = [[1, 1, 1], [2, 2, 1], [3, 3, 1], [1, 3, 0], [2, 3, 0], [3, 4, 0]]
# print(samples)

fig = plt.figure()
fig.canvas.mpl_connect('button_press_event', on_press)
X, y, W = start_train(samples_data)
redraw(X, y, W)


