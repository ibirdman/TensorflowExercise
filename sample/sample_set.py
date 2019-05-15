import numpy as np
import os
import csv
from sklearn.model_selection import train_test_split

class Sample():
    def __init__(self, feature_num):
        self.features = np.zeros(shape=(0, feature_num), dtype=np.float)
        self.labels = np.zeros(shape=(0, 1), dtype=np.int)
        self.start = 0
        
    def next_batch(self, batch_size):
        end = (self.start + batch_size) % self.features.shape[0]
        features = self.features[self.start:end, :]
        labels = self.labels[self.start:end, :]
        self.start = end
        return features, labels;   

class SampleSet():
    def __init__(self, feature_num):
        self.train = Sample(feature_num)
        self.validation = Sample(feature_num)
        self.test = Sample(feature_num)
        self.all = Sample(feature_num)
        
def make_one_hot(data, label_num):
    return (np.arange(label_num) == data[:]).astype(np.int)
        
def load_samples(csv_file, one_hot=False):
    if (os.path.exists(csv_file)):
        with open(csv_file) as f:
            reader = csv.reader(f)
            header = next(reader) #skip the header line
            samples_data = np.zeros(shape=(0, len(header)), dtype=np.float)
            for row in reader:
                sample = np.array(row).astype(np.float)
                samples_data = np.append(samples_data, [sample], axis=0)

    feature_num = len(header) - 1
    sample_set = SampleSet(feature_num)
    if (len(samples_data) > 0):
        np.random.shuffle(samples_data)

        train_size = int(len(samples_data)*7/10)
        train_data = samples_data[0:train_size, :]
        validation_size = int(len(samples_data)*2/10)
        validation_data = samples_data[train_size:train_size + validation_size, :]
        test_size = int(len(samples_data)*1/10)
        test_data = samples_data[train_size + validation_size:train_size + validation_size + test_size, :]

        sample_set.train.features = train_data[:, 0:feature_num]
        sample_set.validation.features = validation_data[:, 0:feature_num]
        sample_set.test.features = test_data[:, 0:feature_num]
        sample_set.all.features = samples_data[:, 0:feature_num]
        if one_hot:
            max_label = int(max(samples_data[:, len(header) - 1]))
            label_num = max_label + 1
            sample_set.train.labels = make_one_hot(train_data[:, feature_num:feature_num + 1], label_num)
            sample_set.validation.labels = make_one_hot(validation_data[:, feature_num:feature_num + 1], label_num)
            sample_set.test.labels = make_one_hot(test_data[:, feature_num:feature_num + 1], label_num)
            sample_set.all.labels = make_one_hot(samples_data[:, feature_num:feature_num + 1], label_num)
        else:
            sample_set.train.labels = train_data[:, feature_num:feature_num + 1].astype(np.int)
            sample_set.validation.labels = validation_data[:, feature_num:feature_num + 1].astype(np.int)
            sample_set.test.labels = test_data[:, feature_num:feature_num + 1].astype(np.int)
            sample_set.all.labels = samples_data[:, feature_num:feature_num + 1].astype(np.int)

    return sample_set;

'''
sample_set = load_samples('../data/melon.csv')
#print(sample_set.all.features, sample_set.all.labels)

x_train,x_test,y_train,y_test = train_test_split(sample_set.all.features,
                                                 sample_set.all.labels,
                                                 test_size = 0.0,
                                                 random_state = 33)  # 为了复现实验，设置一个随机数

print(x_train, '\n', y_train)
'''


