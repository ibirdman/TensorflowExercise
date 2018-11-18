import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

path =  'data' + os.sep + 'scores.txt'
print(path)
pdData = pd.read_csv(path,header = None ,names = ['Exam 1','Exam 2','admin'])
pdData.head()
# print(pdData)

positive = pdData[pdData['admin'] == 1]
negative = pdData[pdData['admin'] == 0]
fix,ax = plt.subplots(figsize = (10,5))
ax.scatter(positive['Exam 1'],positive['Exam 2'],s = 30,c = 'b',marker = 'o',label = 'Admitted')
ax.scatter(negative['Exam 1'],negative['Exam 2'],s = 30,c = 'r',marker = 'x',label = 'Not Admintted')
ax.legend()
ax.set_xlabel('Exam 1 score')
ax.set_ylabel('Exam 2 socre')
plt.show()

def sigmoid(z):
    return 1 / (1+np.exp(-z))

nums = np.arange(-10,10,step = 1)
fig,ax = plt.subplots(figsize = (12,4))
ax.plot(nums,sigmoid(nums),'r')
plt.show()

