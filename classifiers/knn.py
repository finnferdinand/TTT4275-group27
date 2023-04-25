#!/usr/bin/env python
"""
Classifier based on the nearest neighbor decision rule.
It uses the methods described in Johnsen, Magne H.; Classification 
(2017) pp. 10 & 18-20.
"""

import numpy as np
import scipy.spatial
import concurrent.futures
from matplotlib import pyplot as plt

from .classifier import Classifier

__author__ = "Finn Ferdinand Sandvand and Christian Le"
__copyright__ = "Copyright 2023"
__credits__ = ["Magne H. Johnsen"]
__license__ = "MIT"

class NN(Classifier):
    def __init__(self, dataset):
        super().__init__()
        self.dataset          = dataset
        self.confusion_matrix = np.zeros([self.dataset.row_size, self.dataset.col_size])

    def test(self, num_chunks):
        if self.dataset.trainlab.shape[0] % num_chunks != 0:
            raise Exception("num_chunks is not evenly divisible by the number of training samples")
        if  self.dataset.testlab.shape[0] % num_chunks != 0:
            raise Exception("num_chunks is not evenly divisible by the number of test samples")

        print("Testing 1NN classifier. This may take a few seconds...")

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
            )))
        self.confusion_matrix = scipy.stats.contingency.crosstab(self.dataset.testlab, self.classified_labels).count

    def plot_misclassified(self, selection_size):
        misclassified_filter = self.classified_labels != self.dataset.testlab.flatten()
        misclassified_labels = self.classified_labels[misclassified_filter] # all misclassified labels
        misclassified_labels = misclassified_labels[:selection_size]        # selection_size first misclassified labels
        image_data = self.dataset.testv[misclassified_filter, :]            # all misclassified data
        image_data = image_data[:selection_size, :]                         # selection_size first misclassified data
        correct_labels = self.dataset.testlab.flatten()[misclassified_filter]
        correct_labels = correct_labels[:selection_size]
        self._plot_selection(image_data, misclassified_labels, correct_labels)

    def plot_correctly_classified(self, selection_size):
        correctly_classified_filter = self.classified_labels == self.dataset.testlab
        correctly_classified_labels = self.classified_labels[correctly_classified_filter] # all correctly classified labels
        correctly_classified_labels = correctly_classified_labels[:selection_size]        # selection_size first correctly classified labels
        image_data = self.dataset.testv[correctly_classified_filter, :]                   # all correctly classified data
        image_data = image_data[:selection_size, :]                                       # selection_size first correctly classified data
        correct_labels = self.dataset.testlab[correctly_classified_filter]
        correct_labels = correct_labels[:selection_size]
        self._plot_selection(image_data, correctly_classified_labels, correct_labels)

    def _classify_chunk(self, train_subset_data, test_subset_data, train_subset_labels):
        """
        Classifies a chunk of the test data according to the nearest neighbor rule
        on a subset of the training data.
        """
        dist = scipy.spatial.distance_matrix(train_subset_data, test_subset_data) # calculate the distance matrix (distance from train_i to test_j)
        nearest_neighbor_index_array = np.argmin(dist, axis=0)                    # find the indexes of the nearest neighbors (for each test_j, which train_i has shortest distance)
        return np.take(train_subset_labels, nearest_neighbor_index_array)         # classified label is the label of the nearest neighbour 

    def _plot_selection(self, data, classified_labels, correct_labels):
        selection_size = len(data) # must be even
        super().new_figure()
        for index in range(selection_size):
            plt.subplot(2, selection_size//2, index+1)
            square_test_image = np.reshape(data[index], [self.dataset.row_size, self.dataset.col_size])
            plt.imshow(square_test_image, interpolation='nearest')
            plt.title(f"Classified: {classified_labels[index]}\nActual: {correct_labels[index]}")
