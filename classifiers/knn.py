#!/usr/bin/env python
"""
Classifier based on the nearest neighbor decision rule.
It uses the methods described in Johnsen, Magne H.; Classification 
(2017) pp. 10 & 18-20.

The clustering algorithm used is KMeans from scikit-learn. This
algorithm was chosen due to its excellent runtime performance:
https://hdbscan.readthedocs.io/en/latest/performance_and_scalability.html#comparison-of-high-performance-implementations
"""

import numpy as np
import scipy.spatial
import scipy.stats
import concurrent.futures
import time
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

from .classifier import Classifier

__author__ = "Finn Ferdinand Sandvand and Christian Le"
__copyright__ = "Copyright 2023"
__credits__ = ["Magne H. Johnsen"]
__license__ = "MIT"

class kNN(Classifier):
    def __init__(self, dataset):
        super().__init__()
        self.dataset          = dataset
        self.confusion_matrix = np.zeros([self.dataset.row_size, self.dataset.col_size])
        self.num_clusters     = None

    def test(self, num_chunks = 50, k = 1):
        print("Testing knn classifier...")
        start = time.time()
        self.num_chunks = num_chunks
        self.k = k
        extended_k = (np.ones(self.num_chunks)*k).astype(int)
        if self.num_clusters is None: # The classifier has not been trained, use full training data
            if self.dataset.trainlab.shape[0] % num_chunks != 0:
                raise Exception("num_chunks is not evenly divisible by the number of training samples")
            if self.dataset.testlab.shape[0] % num_chunks != 0:
                raise Exception("num_chunks is not evenly divisible by the number of test samples")
    
            # Threads are used to test in parallell to speed up the process.
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_chunks) as executor:
                # executor.map(args) returns a list of classified labels for each chunk in order
                # all classified labels are then collected in a single numpy array by concatenating this list of arrays
                self.classified_labels = np.concatenate(list(executor.map(
                    self._classify_chunk, 
                    np.split(self.dataset.trainv, self.num_chunks), 
                    np.split(self.dataset.testv, self.num_chunks),
                    np.split(self.dataset.trainlab, self.num_chunks),
                    extended_k
                )))
        else:                         # The classifier has been trained so clusters should be used
            extended_c = np.tile(self.centroids, (self.num_chunks, 1))
            extended_clabel = np.tile(self.clabel, self.num_chunks)
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_chunks) as executor:
                self.classified_labels = np.concatenate(list(executor.map(
                    self._classify_chunk, 
                    np.split(extended_c, self.num_chunks),
                    np.split(self.dataset.testv, self.num_chunks),
                    np.split(extended_clabel, self.num_chunks),
                    extended_k
                )))
        self.confusion_matrix = scipy.stats.contingency.crosstab(self.dataset.testlab, self.classified_labels).count
        self.log_write(f"Testing complete after: {round(time.time() - start, 2)} seconds")
    
    def plot_misclassified(self):
        num_extracted_data = 10
        misclassified_filter = self.classified_labels != self.dataset.testlab
        misclassified_labels = self.classified_labels[misclassified_filter] # all misclassified labels
        misclassified_labels = misclassified_labels[:num_extracted_data]    # selection_size first misclassified labels
        image_data = self.dataset.testv[misclassified_filter, :]            # all misclassified data
        image_data = image_data[:num_extracted_data, :]                     # selection_size first misclassified data
        correct_labels = self.dataset.testlab.flatten()[misclassified_filter]
        correct_labels = correct_labels[:num_extracted_data]
        self._plot_selection(image_data, misclassified_labels, correct_labels)
        plt.suptitle(f"Misclassified samples using {self.num_chunks} chunks using {self.k}NN"
                     + (f" with {self.num_clusters}-means clustering per class." if self.num_clusters is not None else "."))
        self.save_figure()

    def plot_correctly_classified(self):
        num_extracted_data = 10
        correctly_classified_filter = self.classified_labels == self.dataset.testlab
        correctly_classified_labels = self.classified_labels[correctly_classified_filter] # all correctly classified labels
        correctly_classified_labels = correctly_classified_labels[:num_extracted_data]    # num_extracted_data first correctly classified labels
        image_data = self.dataset.testv[correctly_classified_filter, :]                   # all correctly classified data
        image_data = image_data[:num_extracted_data, :]                                   # num_extracted_data first correctly classified data
        correct_labels = self.dataset.testlab[correctly_classified_filter]
        correct_labels = correct_labels[:num_extracted_data]
        self._plot_selection(image_data, correctly_classified_labels, correct_labels)
        plt.suptitle(f"Correctly classified samples using {self.num_chunks} chunks using {self.k}NN"
                     + (f" with {self.num_clusters}-means clustering per class." if self.num_clusters is not None else "."))
        self.save_figure()

    def train(self, num_clusters):
        """
        Clusters each class to specified number of clusters.
        Returns the corresponding centroid and it's label.
        """
        start = time.time()
        print("Training kNN classifier....")
        self.num_clusters = num_clusters
        self.clabel = np.repeat(np.arange(self.dataset.num_classes), self.num_clusters)
        self.centroids = np.empty((self.dataset.num_classes * self.num_clusters, self.dataset.vec_size), dtype=int)
        for digit_class in range(self.dataset.num_classes):
            class_filter = digit_class == self.dataset.trainlab
            data_from_class = self.dataset.trainv[class_filter, :]
            kmeans = KMeans(n_clusters=self.num_clusters, n_init=1).fit(data_from_class)
            self.centroids[digit_class*num_clusters:(digit_class+1)*num_clusters,:] = kmeans.cluster_centers_.astype(int)
        self.log_write(f"Clustering complete after: {round(time.time() - start, 2)} seconds\n")

    def _classify_chunk(self, train_subset_data, test_subset_data, train_subset_labels, k):
        """
        Classifies a chunk of the test data according to the 
        nearest neighbor rule on a subset of the training data.
        """
        dist = scipy.spatial.distance_matrix(train_subset_data, test_subset_data) # calculate the distance matrix
        k_nearest_indexes = np.argpartition(dist, k, axis=0)[:k]                  # indexes of the nearest neighbors
        k_nearest_labels = train_subset_labels[k_nearest_indexes]                 # labels of the nearest neighbors
        return scipy.stats.mode(k_nearest_labels, keepdims=False).mode            # classified label is the most common label (mode)

    def _plot_selection(self, data, classified_labels, correct_labels):
        """
        Plots the selected data and their classified labels along 
        with the correct label.
        """
        selection_size = len(data)
        self.new_figure()
        for index in range(selection_size):
            plt.subplot(2, selection_size//2, index+1)
            square_test_image = np.reshape(data[index], [self.dataset.row_size, self.dataset.col_size])
            plt.imshow(square_test_image, interpolation='nearest')
            plt.title(f"Classified: {classified_labels[index]}\nActual: {correct_labels[index]}")
