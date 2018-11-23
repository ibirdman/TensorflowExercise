import numpy as np

W = [0.3, 0.1, 0.2] # 线性参数列表 (W[0] is bias)
DIM = len(W) - 1       # 属性数量

# sample point count for training.
m = 30
x_data = np.float32(np.random.rand(m, DIM)) # 随机输入
w_data = np.array(W[1:]).reshape(DIM, 1)
y_data = np.dot(x_data, w_data) + W[0]
# print(x_data)
# print(y_data)

# Points x-coordinate and dummy value (x0, x1).
X = np.hstack((np.ones((m, 1)), x_data)).reshape(m, DIM + 1)
# Points y-coordinate
y = y_data.reshape(m, 1)

# The Learning Rate alpha.
alpha = 0.5

def error_function(theta, X, y):
    '''Error function J definition.'''
    diff = np.dot(X, theta) - y
    return (1./2*m) * np.dot(np.transpose(diff), diff)

def gradient_function(theta, X, y):
    '''Gradient of the function J definition.'''
    diff = np.dot(X, theta) - y
    return (1./m) * np.dot(np.transpose(X), diff)

def gradient_descent(X, y, alpha):
    '''Perform gradient descent.'''
    theta = np.ones((DIM + 1, 1))
    gradient = gradient_function(theta, X, y)
    count = 0
    while not np.all(np.absolute(gradient) <= 1e-6):
        theta = theta - alpha * gradient
        gradient = gradient_function(theta, X, y)
        count = count + 1
        
    print("loop=" + str(count))
    return theta

optimal = gradient_descent(X, y, alpha)
print('optimal:\n', optimal)
print('error function:', error_function(optimal, X, y)[0,0])

