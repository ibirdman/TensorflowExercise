import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import csv

class Sample():
    def __init__(self):
        self.images = np.zeros(shape=(0, 2), dtype=np.float)
        self.labels = np.zeros(shape=(0, 1), dtype=np.int)
        self.start = 0
        
    def next_batch(self, batch_size):
        end = (self.start + batch_size) % self.images.shape[0]
        images = self.images[self.start:end, :]
        labels = self.labels[self.start:end, :]
        self.start = end
        return images, labels;   

class SampleSet():
    def __init__(self):
        self.train = Sample() 
        self.validation = Sample()
        self.test = Sample()
        
def make_one_hot(data, label_num):
    return (np.arange(label_num) == data[:]).astype(np.int)
        
def load_samples(csv_file, one_hot=True):
    if (os.path.exists(csv_file)):
        with open(csv_file) as f:
            reader = csv.reader(f)
            header = next(reader) #skip the header line
            samples_data = np.zeros(shape=(0, len(header)), dtype=np.float)
            for row in reader:
                sample = np.array(row).astype(np.float)
                samples_data = np.append(samples_data, [sample], axis=0)

    sample_set = SampleSet()
    if (len(samples_data) > 0):
        np.random.shuffle(samples_data)

        train_size = int(len(samples_data)*7/10);
        train_data = samples_data[0:train_size,:]
        validation_size = int(len(samples_data)*2/10);
        validation_data = samples_data[train_size:train_size + validation_size,:]
        test_size = int(len(samples_data)*1/10);
        test_data = samples_data[train_size + validation_size:train_size + validation_size + test_size,:]
        
        sample_set.train.images = train_data[:,0:2]
        sample_set.validation.images = validation_data[:,0:2]
        sample_set.test.images = test_data[:,0:2]
        if one_hot:
            max_label = int(max(samples_data[:,2]))   
            label_num = max_label + 1
            sample_set.train.labels = make_one_hot(train_data[:,2:3], label_num)
            sample_set.validation.labels = make_one_hot(validation_data[:,2:3], label_num)
            sample_set.test.labels = make_one_hot(test_data[:,2:3], label_num)            
        else:
            sample_set.train.labels = train_data[:,2:3].astype(np.int)
            sample_set.validation.labels = validation_data[:,2:3].astype(np.int)
            sample_set.test.labels = test_data[:,2:3].astype(np.int)

    return sample_set;

#sample_set = load_samples('../data/mysamples4.csv')
#print(sample_set.validation.labels)


