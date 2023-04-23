#!/usr/bin/env python

from classifiers.knn import NN
from datasets.mnistdataset import MNISTDataSet

__author__ = "Finn Ferdinand Sandvand and Christian Le"
__copyright__ = "Copyright 2023"
__license__ = "MIT"

digits_data_path = 'datasets/data/MNist_ttt4275/data_all.mat'

def digits():
    print("\n\n------------------------ DIGITS ------------------------")
    print("Using full traning set as templates")
    nn_classifier = NN(MNISTDataSet(digits_data_path))
    nn_classifier.test(num_chunks=50)
    nn_classifier.print_performance()
    # plot a selection of misclassified and correctly classified test samples
    selection_size = 10
    nn_classifier.plot_misclassified(selection_size)
    nn_classifier.plot_correctly_classified(selection_size)

    # TODO: KNN classifier
