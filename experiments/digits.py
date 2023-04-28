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
    nn_classifier.log_write("\n\n-------------------------- DIGITS --------------------------\n")  

    # Task 1a NN classifier using Euclidean distance and chunks
    # Task 1b&c (Plot classified and misclassified images)
    print("\nUsing kNN classifier with num_chunks = 50")
    nn_classifier.test(num_chunks = 50)
    nn_classifier.log_performance("num_chunks = 50") 
    nn_classifier.plot_misclassified()
    nn_classifier.plot_correctly_classified()

    # Task 2a&b (Create 64 clusters per class and check significance of these templates)
    print("\nUsing kNN classifier with num_chunks = 50, num_clusters = 64, k = 1")
    nn_classifier.train(num_clusters = 64)
    nn_classifier.test(num_chunks=50, k=1)
    nn_classifier.log_performance("num_chunks = 50, num_clusters = 64, k = 1")

    # Task 2c (Test significance of 'k'NN neighours)
    print("\nUsing kNN classifier with num_chunks = 50, num_clusters = 64, k = 7")
    nn_classifier.test(num_chunks=50, k=7)
    nn_classifier.log_performance("num_chunks = 50, num_clusters = 64, k = 7")
