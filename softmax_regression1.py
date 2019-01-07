import numpy as np
import matplotlib.pyplot as plt

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

def load_data():
     digits = load_digits()
     data = digits.data
     label = digits.target
     return np.mat(data), label

def gradient_descent(train_x, train_y, k, maxCycle, alpha):
# k 为类别数
     numSamples, numFeatures = np.shape(train_x)
     weights = np.mat(np.ones((numFeatures, k)))
     
     for i in range(maxCycle):
          value = np.exp(train_x * weights)  
          rowsum = value.sum(axis = 1)   # 横向求和
          rowsum = rowsum.repeat(k, axis = 1)  # 横向复制扩展
          err = - value / rowsum  #计算出每个样本属于每个类别的概率
          for j in range(numSamples):     
               err[j, train_y[j]] += 1
          weights = weights + (alpha / numSamples) * (train_x.T * err)
     return weights
    
     

def test_model(test_x, test_y, weights):
     results = test_x * weights
     predict_y = results.argmax(axis = 1)
     count = 0
     for i in range(np.shape(test_y)[0]):
          if predict_y[i,] == test_y[i,]:
               count += 1   
     return count / len(test_y), predict_y 


def on_press(event):
    global samples_data
    # print("my position:" ,event.button,event.xdata, event.ydata)
    if event.button == 1: cla = 1
    elif event.button == 3: cla = 0
    samples_data.append([event.xdata,event.ydata,cla])  
    # print(samples_data)
    X, y, W = start_train(samples_data)
    plt.cla()
    redraw(X, y, W)
    
def start_train(sample_data):
    samples = np.array(samples_data)
    train_x = samples[:,0:2]
    one = np.ones((train_x.shape[0], 1))
    train_x = np.hstack((one, train_x))
    train_y = samples[:,2].reshape(-1,1)
    print(train_x)
    print(train_y) 
    k = len(np.unique(train_y))
    print(k)
    weights = gradient_descent(train_x, train_y, k, 8000, 0.01)
    print(weights) 
    return train_x, train_y, weights

def redraw(X, y, W):
    label = np.array(y)
    index_0 = np.where(label==0)
    plt.scatter(X[index_0,1],X[index_0,2],marker='x',color = 'b',label = '0',s = 15)
    index_1 =np.where(label==1)
    plt.scatter(X[index_1,1],X[index_1,2],marker='o',color = 'r',label = '1',s = 15)
    
    
    bv = np.array([1.,2.,3.])
    for i in range(bv.shape[0]):
        bv[i] = W[i,0]

    print("bv--")
    print(bv)
    W = bv
    

    
    num = W.shape[0]
    for l in range(num):
        #show the decision boundary
        x1 = np.arange(0,10,0.5)
        test = np.array(W[-1,l])
        print(test)
        x2 = (- W[0,l] - W[1,l]*x1) / W[2,l]
        print("....")
        print(W)
        print(x1)
        print(x2)
        plt.plot(x1,x2,color = 'black')


    plt.legend(loc = 'upper left')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.draw()
    
samples_data = [[1, 1, 1], [2, 2, 1], [3, 3, 1], [1, 2, 0], [2, 3, 0], [3, 4, 0]]
# print(samples)

fig = plt.figure()
fig.canvas.mpl_connect('mouse_press_event', on_press)
X, y, W = start_train(samples_data)
redraw(X, y, W)

plt.show()
