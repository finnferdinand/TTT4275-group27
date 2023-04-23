#!/usr/bin/env python

import scipy.io

__author__ = "Finn Ferdinand Sandvand and Christian Le"
__copyright__ = "Copyright 2023"
__license__ = "MIT"

class MNISTDataSet(object):
    def __init__(self, data_path):
        data_all = scipy.io.loadmat(data_path)
        self.trainv   = data_all['trainv']             # training data
        self.trainlab = data_all['trainlab'].flatten() # training labels
        self.testv    = data_all['testv']              # test data
        self.testlab  = data_all['testlab'].flatten()  # test labels

        self.num_train = data_all['num_train'][0,0]    # number of training samples
        self.num_test  = data_all['num_test'][0,0]     # number of testing samples
        self.row_size  = data_all['row_size'][0,0]     # number of rows in a single image
        self.col_size  = data_all['col_size'][0,0]     # number of columns in a single image
        self.vec_size  = data_all['vec_size'][0,0]     # number of elements in an image vector
