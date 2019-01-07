import numpy as np
import matplotlib.pyplot as plt
import os
import csv

class logistic(object):
    def __init__(self):
        self.W = None
    def train(self,X,y,learn_rate = 0.01,num_iters = 1000):
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
    
def on_key_press(event):
    print(event.key)
    if event.key == 'ctrl+z':
        X, y, W = start_train()
        redraw(X, y, W)
    
def start_train(classify):
    samples_data = load_samples('data/mysamples2.csv')
    print(samples_data)

    samples = np.array(samples_data)
    X = samples[:,0:2]
    y = samples[:,2:3]
    # print(X)
    # print(y)    
     
    y = y.reshape((-1,1))
    #add the x0=1
    one = np.ones((X.shape[0],1))
    X_train = np.hstack((one,X))
    loss = classify.train(X_train,y)
    print(classify.W)
    return X, y, classify.W

def redraw(X, y, W):
    plt.clf()

    draw_training_data(X, y, W)
    draw_predict_map(X, y, W)
    
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend(loc = 'upper left')
    plt.show()

def load_samples(csv_file):
    samples_data = np.zeros(shape=(0, 3), dtype=np.float)
    if (os.path.exists(csv_file)):
        with open(csv_file) as f:
            reader = csv.reader(f) #
            next(reader) #skip the header line
            for row in reader:
                sample = np.array(row)
                samples_data = np.append(samples_data, [sample.astype(np.float)], axis=0)            
    return samples_data;
    
def draw_training_data(X, y, W):
    label = np.array(y)
    index_0 = np.where(label==0)[0]
    plt.scatter(X[index_0,0],X[index_0,1],marker='x',color = 'b',label = '0',s = 15)
    index_1 =np.where(label==1)[0]
    plt.scatter(X[index_1,0],X[index_1,1],marker='o',color = 'r',label = '1',s = 15)   
    
def draw_predict_map(X, y, W):
    global logistic_classify
    x1_range = np.array([min(X[:,0]), max(X[:,0])], dtype=np.float)
    x2_range = np.array([min(X[:,1]), max(X[:,1])], dtype=np.float)
    X1 = np.arange(x1_range[0], x1_range[1], (x1_range[1] - x1_range[0]) / 20)
    X2 = np.arange(x2_range[0], x2_range[1], (x2_range[1] - x2_range[0]) / 20)
    for x1 in X1:
        for x2 in X2:
           x_verify = np.array([1, x1, x2], dtype=np.float)
           y_predict = logistic_classify.predict(x_verify)
           plt.scatter(x1, x2, c = np.where(y_predict == 0, 'blue', 'red'), alpha = 0.2, edgecolors='none', s=10)
  
# start
logistic_classify = logistic()
X, y, W = start_train(logistic_classify)

fig = plt.figure()
fig.canvas.mpl_connect('key_press_event', on_key_press)
redraw(X, y, W)


