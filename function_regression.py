import numpy as np

# Size of the points dataset.
m = 20

# Points x-coordinate and dummy value (x0, x1).
X0 = np.ones((m, 1))
X1 = np.arange(1, m+1).reshape(m, 1)
# print(X1)
X = np.hstack((X0, X1))
# print(X)

theta0 = 1
step = 0.8

def func(x):
    return (x - 3)**2 + 5
    
def derivative_func(x):
    return 2*(x - 3)

theta = [theta0]
for i in range(20):
    print("[" + str(i) + "]=" + str(theta[-1]))
    x = theta[-1]
    delta = -derivative_func(x)*step
    theta.append(x + delta)

print("x=" + str(theta[-1]))
print("J=" + str(func(theta[-1])))

