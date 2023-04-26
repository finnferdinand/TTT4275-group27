#!/usr/bin/env python

import numpy as np

__author__ = "Finn Ferdinand Sandvand and Christian Le"
__copyright__ = "Copyright 2023"
__license__ = "MIT"

class IrisDataSet(object):
    def __init__(self, data_path, train_test_ratio):
        self.data            = np.loadtxt(data_path, delimiter=',', usecols=[0,1,2,3])
        self.num_data_points = self.data.shape[0]
        self.num_features    = self.data.shape[1]
        labels               = np.loadtxt(data_path, delimiter=',', usecols=[4], dtype='|S15')
        self.num_classes     = len(np.unique(labels))
        self.num_train       = int(self.num_data_points / self.num_classes * train_test_ratio)
        self.num_test        = int(self.num_data_points / self.num_classes * (1-train_test_ratio))
        self.per_class       = self.num_train + self.num_test

    def partition_dataset(self, training_first = True):
        if training_first: # select first subset as training set and last as test set
            self.trainv  = self.data[[row % self.per_class < self.num_train for row in range(self.num_data_points)],:]
            self.testv   = self.data[[row % self.per_class >= self.num_train for row in range(self.num_data_points)],:]
        else:              # select first subset as test set and last as training set
            self.testv   = self.data[[row % self.per_class < self.num_test for row in range(self.num_data_points)],:]
            self.trainv  = self.data[[row % self.per_class >= self.num_test for row in range(self.num_data_points)],:]
        self.full_trainv = self.trainv
        self.full_testv  = self.testv

    def select_features(self, features):
        self.trainv       = self.full_trainv[:, features]
        self.testv        = self.full_testv[:, features]
        self.num_features = len(features)
