import numpy as np

class IrisDataSet(object):
        def __init__(self, data_path, train_test_ratio):
            self.data                   = np.loadtxt(data_path, delimiter=',', usecols=[0,1,2,3])
            self.num_data_points        = self.data.shape[0]
            self.num_features           = self.data.shape[1]
            labels                      = np.loadtxt(data_path, delimiter=',', usecols=[4], dtype='|S15')
            self.num_classes            = len(np.unique(labels))
            self.train_confusion_matrix = np.zeros([self.num_classes, self.num_classes])
            self.num_train              = int(self.num_data_points / self.num_classes * train_test_ratio)
            self.num_test               = int(self.num_data_points / self.num_classes * (1-train_test_ratio))
            self.per_class              = self.num_train + self.num_test

        def partition_dataset(self, training_first = True):
            if training_first: # select first subset as training set and last as test set
                self.trainv = self.data[[row % self.per_class < self.num_train for row in range(self.num_data_points)],:]
                self.testv  = self.data[[row % self.per_class >= self.num_train for row in range(self.num_data_points)],:]
            else:              # select first subset as test set and last as training set
                self.testv  = self.data[[row % self.per_class < self.num_test for row in range(self.num_data_points)],:]
                self.trainv = self.data[[row % self.per_class >= self.num_test for row in range(self.num_data_points)],:]
