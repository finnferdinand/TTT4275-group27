#!/usr/bin/env python
"""
Classifier based on a linear decision rule, using gradient descent
training with a minimum mean square error cost function.
It uses the methods described in Johnson, Magne H.; Classification 
(2017) pp. 9-10 & 15-18.

Equations are implemented as described in Johnson, however, in
order to speed up computation time and for shorter notation
the vectors g_k, t_k, and x_k have been converted into the matrices
g, t, and x containing the values for all k. This avoids using
sums and for loops many places, which are computationally expensive.
"""

import numpy as np
from matplotlib import pyplot as plt
import scipy.stats.contingency

from .classifier import Classifier

__author__ = "Finn Ferdinand Sandvand and Christian Le"
__copyright__ = "Copyright 2023"
__credits__ = ["Magne H. Johnson"]

class Linear(Classifier):
    class Configuration(object):
        def __init__(self, step_size, max_iterations, threshold):
            self.step_size        = step_size      # alpha as in eq (23) in Johnson
            self.max_iterations   = max_iterations # maximum number of iterations
            self.threshold        = threshold      # terminate training if ||grad MSE|| < this number

    def __init__(self, dataset, configuration):
        super().__init__()
        self.configuration          = configuration
        self.dataset                = dataset
        self.train_confusion_matrix = np.zeros([self.dataset.num_classes, self.dataset.num_classes])
        self.test_confusion_matrix  = np.zeros([self.dataset.num_classes, self.dataset.num_classes])
        self.mse                    = []

    def train(self, training_first = True, selected_features = None):
        print("Training linear classifier...")
        self.dataset.partition_dataset(training_first)     
        if selected_features is not None:
            self.dataset.select_features(selected_features)   

        t = np.kron(np.eye(self.dataset.num_classes), np.ones(self.dataset.num_train))
        x = np.hstack((self.dataset.trainv, 
                       np.ones((self.dataset.num_train * self.dataset.num_classes, 1))
                       )).transpose()                                                  # transformation [x^T 1]^T -> x
        self.W = np.zeros([self.dataset.num_classes, self.dataset.num_features + 1])   # initialize as (C,D+1) zero matrix
        self.mse = []

        terminating_criteria = False
        iteration = 0
        while not terminating_criteria:
            z = self.W @ x
            g = self._sigmoid(z)                                      # eq (20) in Johnson
            gradient = self._MSEgradient(g, t, x)
            self.W = self.W - self.configuration.step_size * gradient # eq (23) in Johnson
            iteration += 1
            mse = 0.5 * np.sum((g - t).T @ (g - t), axis=1).sum()       # eq (19) in Johnson
            self.mse.append(mse)
            terminating_criteria = iteration > self.configuration.max_iterations \
                                 or np.linalg.norm(gradient) < self.configuration.threshold

        print(f"Training terminated at iteration: {iteration-1}, with ||grad MSE|| = {round(np.linalg.norm(gradient),2)}")
        self.train_confusion_matrix = scipy.stats.contingency.crosstab(np.argmax(g, axis=0), np.argmax(t, axis=0)).count

    def test(self):
        print("Testing linear classifier...")
        t = np.kron(np.eye(self.dataset.num_classes), np.ones(self.dataset.num_test))
        x = np.hstack((self.dataset.testv, 
                       np.ones((self.dataset.num_test * self.dataset.num_classes, 1))
                       )).transpose()                                       # transformation [x^T 1]^T -> x
        z = self.W @ x
        g = self._sigmoid(z)
        self.test_confusion_matrix = scipy.stats.contingency.crosstab(np.argmax(g, axis=0), np.argmax(t, axis=0)).count

    def print_performance(self):
        print("\n~~ PERFORMANCE ~~")
        print("Training Set CM" + self.dataset.num_classes * "   " + "  " + "Test Set CM")
        for c in range(self.dataset.num_classes):
            print(f"{self.train_confusion_matrix[c,:]}" + "   " * self.dataset.num_classes +
                  "       " + f"{self.test_confusion_matrix[c,:]}")
        print(f"Det. rate: {round(self.get_detection_rate(self.train_confusion_matrix)*100, 2)}%" +
              f"   " * self.dataset.num_classes +
              f"Det. rate: {round(self.get_detection_rate(self.test_confusion_matrix)*100, 2)}%")
    
    def plot_histograms(self):
        print("Producing histograms...")
        super().new_figure()
        counter = 0
        for c in range(self.dataset.num_classes):
            for feature in range(self.dataset.num_features):
                counter += 1
                plt.subplot(self.dataset.num_classes, self.dataset.num_features, counter)
                plt.hist(self.dataset.data[c*self.dataset.per_class:(c+1)*self.dataset.per_class, feature], bins=20, range=[0,8])
                if c == 0: plt.title("Feature: " + str(feature))
                if feature == 0: plt.ylabel("Class: " + str(c))

    def plot_MSE(self):
        super().new_figure()
        plt.plot(self.mse)
        plt.title("MSE")

    # helper functions for training
    def _MSEgradient(self, g, t, x):
        return (g - t) * g * (1 - g) @ x.T  # eq (22) in Johnson, rewritten with g, t, and x matrices 
                                            # for significantly shorter comutation time

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))         # eq (20) in Johnson
