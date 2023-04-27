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

    def test(self, num_chunks=10, num_clusters = None, k=1):
        if self.dataset.trainlab.shape[0] % num_chunks != 0 and num_clusters is None:
            raise Exception("num_chunks is not evenly divisible by the number of training samples")
        if self.dataset.testlab.shape[0] % num_chunks != 0 and num_clusters is None:
            raise Exception("num_chunks is not evenly divisible by the number of test samples")
        print("Testing " + str(k) + "NN classifier using " + str(num_chunks) + " chunks" 
              + (" and " + str(num_clusters) + "-means clustering per class" if num_clusters is not None else ("")) 
              + ". This may take a few seconds...\n")
        
        self.num_chunks = num_chunks
        self.num_clusters = num_clusters
        self.k = k

        start = time.time()
        extended_k = (np.ones(num_chunks)*k).astype(int)

        if num_clusters is None:
        # each chunk of test samples can be classified in parallell as the sets are independent
        # therefore, threads are used to test in parallell to speed up the process.
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_chunks) as executor:
                # executor.map(args) returns a list of classified labels for each chunk in order
                # all classified labels are then collected in a single numpy array by concatenating this list of arrays
                self.classified_labels = np.concatenate(list(executor.map(
                    self._classify_chunk, 
                    np.split(self.dataset.trainv, num_chunks), 
                    np.split(self.dataset.testv, num_chunks),
                    np.split(self.dataset.trainlab, num_chunks),
                    extended_k
                )))
        else:
            self._clustering(num_clusters)

            extended_C = np.kron(np.ones((num_chunks, 1)), self.C).astype(int)
            extended_clabel = np.kron(np.ones(num_chunks), self.clabel).astype(int)

            with concurrent.futures.ThreadPoolExecutor(max_workers=num_chunks) as executor:
                self.classified_labels = np.concatenate(list(executor.map(
                    self._classify_chunk, 
                    np.split(extended_C, num_chunks),
                    np.split(self.dataset.testv, num_chunks),
                    np.split(extended_clabel, num_chunks),
                    extended_k
                )))
        self.confusion_matrix = scipy.stats.contingency.crosstab(self.dataset.testlab, self.classified_labels).count
        end = time.time()
        self.log_write(f"Testing complete after: {round(end - start, 2)} seconds")
    
    def plot_misclassified(self):
        num_extracted_data = 10
        misclassified_filter = self.classified_labels != self.dataset.testlab
        misclassified_labels = self.classified_labels[misclassified_filter] # all misclassified labels
        misclassified_labels = misclassified_labels[:num_extracted_data]        # selection_size first misclassified labels
        image_data = self.dataset.testv[misclassified_filter, :]            # all misclassified data
        image_data = image_data[:num_extracted_data, :]                         # selection_size first misclassified data
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
        correctly_classified_labels = correctly_classified_labels[:num_extracted_data]        # num_extracted_data first correctly classified labels
        image_data = self.dataset.testv[correctly_classified_filter, :]                   # all correctly classified data
        image_data = image_data[:num_extracted_data, :]                                       # num_extracted_data first correctly classified data
        correct_labels = self.dataset.testlab[correctly_classified_filter]
        correct_labels = correct_labels[:num_extracted_data]
        self._plot_selection(image_data, correctly_classified_labels, correct_labels)
        plt.suptitle(f"Correctly classified samples using {self.num_chunks} chunks using {self.k}NN"
                     + (f" with {self.num_clusters}-means clustering per class." if self.num_clusters is not None else "."))
        self.save_figure()

    def _clustering(self, num_clusters):
        self.clabel = np.empty((1, 0), dtype=int)
        self.C = np.empty((0, self.dataset.vec_size), dtype=int)
        for c in range(self.dataset.num_classes):
            class_filter = c == self.dataset.trainlab
            data_from_class = self.dataset.trainv[class_filter, :]
            kmeans = KMeans(n_clusters=num_clusters, n_init=1).fit(data_from_class)
            self.clabel = np.append(self.clabel, c * np.ones(num_clusters).astype(int))
            self.C = np.vstack((self.C, kmeans.cluster_centers_.astype(int)))

    def _classify_chunk(self, train_subset_data, test_subset_data, train_subset_labels, k):
        """
        Classifies a chunk of the test data according to the nearest neighbor rule
        on a subset of the training data.
        """
        dist = scipy.spatial.distance_matrix(train_subset_data, test_subset_data) # calculate the distance matrix
        k_nearest_indexes = np.argpartition(dist, k, axis=0)[:k]
        k_nearest_labels = train_subset_labels[k_nearest_indexes]
        return scipy.stats.mode(k_nearest_labels, keepdims=False).mode

    def _plot_selection(self, data, classified_labels, correct_labels):
        selection_size = len(data) # must be even
        super().new_figure()
        for index in range(selection_size):
            plt.subplot(2, selection_size//2, index+1)
            square_test_image = np.reshape(data[index], [self.dataset.row_size, self.dataset.col_size])
            plt.imshow(square_test_image, interpolation='nearest')
            plt.title(f"Classified: {classified_labels[index]}\nActual: {correct_labels[index]}")
