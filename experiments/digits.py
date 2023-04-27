#!/usr/bin/env python

from classifiers.knn import kNN
from datasets.mnistdataset import MNISTDataSet

__author__ = "Finn Ferdinand Sandvand and Christian Le"
__copyright__ = "Copyright 2023"
__license__ = "MIT"

digits_data_path = 'datasets/data/MNist_ttt4275/data_all.mat'

def digits():
    print("\n\n-------------------------- DIGITS --------------------------")
    print(".: Using full traning set as templates :.")
    nn_classifier = kNN(MNISTDataSet(digits_data_path))

    # Task 1a (Test significance of chunks)
    # Task 1b&c (Plot classified and misclassified images)
    nn_classifier.test(num_chunks=1000)
    nn_classifier.log_performance("num_chunks = 1000") 
    nn_classifier.plot_misclassified()
    nn_classifier.plot_correctly_classified()

    nn_classifier.test(num_chunks=10)
    nn_classifier.log_performance("num_chunks = 10") 
    nn_classifier.plot_misclassified()
    nn_classifier.plot_correctly_classified()

    # Task 2a&b (Create 64 clusters per class and check significance of these templates)
    nn_classifier.test(num_chunks=1000, num_clusters=64, k=1)
    nn_classifier.log_performance("num_chunks = 1000, num_clusters = 64, k = 1")
    nn_classifier.plot_misclassified()
    nn_classifier.plot_correctly_classified()

    # Task 2c (Test significance of 'k'NN neighours)
    nn_classifier.test(num_chunks=10, num_clusters=64, k=3)
    nn_classifier.log_performance("num_chunks = 10, num_clusters = 64, k = 3")
    nn_classifier.plot_misclassified()
    nn_classifier.plot_correctly_classified()

    nn_classifier.test(num_chunks=10, num_clusters=64, k=5)
    nn_classifier.log_performance("num_chunks = 10, num_clusters = 64, k = 5")
    nn_classifier.plot_misclassified()
    nn_classifier.plot_correctly_classified()

    nn_classifier.test(num_chunks=10, num_clusters=64, k=7)
    nn_classifier.log_performance("num_chunks = 10, num_clusters = 64, k = 7")
    nn_classifier.plot_misclassified()
    nn_classifier.plot_correctly_classified()
