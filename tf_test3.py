import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import csv

class Sample():
    def __init__(self):
        self.images = np.zeros(shape=(0, 2), dtype=np.float)
        self.labels = np.zeros(shape=(0, 1), dtype=np.float)

class SampleSet():
    def __init__(self):
        self.train = Sample() 
        self.validation = Sample()
        self.test = Sample()
       
class SampleManager():
    def __init__(self):
        self.sample_set = SampleSet()   
        
    def load(self, csv_file):
        samples_data = np.zeros(shape=(0, 3), dtype=np.float)
        if (os.path.exists(csv_file)):
            with open(csv_file) as f:
                reader = csv.reader(f) #
                next(reader) #skip the header line
                for row in reader:
                    sample = np.array(row)
                    samples_data = np.append(samples_data, [sample.astype(np.float)], axis=0)
    
        if (len(samples_data) > 0):
            np.random.shuffle(samples_data)
            train_size = int(len(samples_data)*7/10);
            print(train_size)
            train_data = samples_data[0:train_size,:]
            validation_size = int(len(samples_data)*2/10);
            validation_data = samples_data[train_size:train_size + validation_size,:]
            test_size = int(len(samples_data)*1/10);
            test_data = samples_data[train_size + validation_size:train_size + validation_size + test_size,:]

            self.sample_set.train.images = train_data[:,0:2]
            self.sample_set.train.labels = train_data[:,2:3]
        return self.sample_set;
        
sm = SampleManager()
sample_set = sm.load('data/mysamples3.csv')
samples_for_check = np.hstack((sample_set.train.images, sample_set.train.labels))
print(samples_for_check)


