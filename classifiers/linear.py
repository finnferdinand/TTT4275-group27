#!/usr/bin/env python
"""
Classifier based on a linear decision rule, using gradient descent
training with a minimum mean square error cost function.
It uses the methods described in Johnsen, Magne H.; Classification 
(2017) pp. 9-10 & 15-18.

Equations are implemented as described in Johnsen, however, in
order to speed up computation time and for shorter notation the 
vectors g_k, t_k, and x_k have been converted into the matrices
G, T, and X containing the values for all training samples. This 
avoids using sums and for loops many places, which are computationally 
expensive.
"""

import numpy as np
from matplotlib import pyplot as plt
import scipy.stats.contingency

from .classifier import Classifier

__author__ = "Finn Ferdinand Sandvand and Christian Le"
__copyright__ = "Copyright 2023"
__credits__ = ["Magne H. Johnsen"]
__license__ = "MIT"

class Linear(Classifier):
    class Configuration(object):
        def __init__(self, step_size, max_iterations, threshold):
            self.step_size      = step_size      # alpha as in eq (23) in Johnsen
            self.max_iterations = max_iterations # maximum number of iterations
            self.threshold      = threshold      # terminate training if ||grad MSE|| < this number

    def __init__(self, dataset, configuration):
        super().__init__()
        self.configuration          = configuration
        self.dataset                = dataset
        self.train_confusion_matrix = np.zeros([self.dataset.num_classes, self.dataset.num_classes])
        self.test_confusion_matrix  = np.zeros([self.dataset.num_classes, self.dataset.num_classes])
        self.mean_square_error      = []

    def train(self, training_first = True, selected_features = None):
        print("Training linear classifier...")
        self.dataset.partition_dataset(training_first)     
        if selected_features is not None:
            self.dataset.select_features(selected_features)   

        T = np.kron(np.eye(self.dataset.num_classes), np.ones(self.dataset.num_train)) # template vector
        X = np.hstack((self.dataset.trainv, 
                       np.ones((self.dataset.num_train * self.dataset.num_classes, 1)) # 'trainv' is the equivalent to x^T
                       )).transpose()                                                  # transformation [x^T 1]^T -> x
        self.W = np.zeros([self.dataset.num_classes, self.dataset.num_features + 1])   # initialize as (C,D+1) zero matrix
        self.mean_square_error = []

        iteration = 0
        terminating_criteria = False
        while not terminating_criteria:
            Z = self.W @ X
            G = self._sigmoid(Z)                                      # eq (20) in Johnsen
            gradient = self._MSEgradient(G, T, X)
            self.W = self.W - self.configuration.step_size * gradient # eq (23) in Johnsen
            mse = 0.5 * np.linalg.norm(G - T)                         # eq (19) in Johnsen
            self.mean_square_error.append(mse)
            iteration += 1
            terminating_criteria = iteration > self.configuration.max_iterations \
                                 or np.linalg.norm(gradient) < self.configuration.threshold

        print(f"Training terminated at iteration: {iteration-1}, with ||grad MSE|| = {round(np.linalg.norm(gradient),2)}")
        self.train_confusion_matrix = scipy.stats.contingency.crosstab(np.argmax(G, axis=0), np.argmax(T, axis=0)).count

    def test(self):
        print("Testing linear classifier...")
        T = np.kron(np.eye(self.dataset.num_classes), np.ones(self.dataset.num_test)) # template vector
        X = np.hstack((self.dataset.testv, 
                       np.ones((self.dataset.num_test * self.dataset.num_classes, 1))
                       )).transpose()                                                 # transformation [x^T 1]^T -> x
        Z = self.W @ X
        G = self._sigmoid(Z)
        self.test_confusion_matrix = scipy.stats.contingency.crosstab(np.argmax(G, axis=0), np.argmax(T, axis=0)).count

    def log_performance(self, title):
        print("Logging performance...")
        self.logger.write("\n" + title + "\n")
        self.log_write("Training Set CM" + self.dataset.num_classes * "   " + "  " + "Test Set CM\n")
        for c in range(self.dataset.num_classes):
            self.log_write(f"{self.train_confusion_matrix[c,:]}" + "   " * self.dataset.num_classes +
                  "       " + f"{self.test_confusion_matrix[c,:]}\n")
        self.log_write(f"Err. rate: {round(self.get_error_rate(self.train_confusion_matrix)*100, 2)}%" +
              f"   " * self.dataset.num_classes +
              f" Err. rate: {round(self.get_error_rate(self.test_confusion_matrix)*100, 2)}%\n")
    
    def plot_histograms(self):
        print("Producing histograms...")
        self.new_figure()
        counter = 0
        for c in range(self.dataset.num_classes):
            for feature in range(self.dataset.num_features):
                counter += 1
                plt.subplot(self.dataset.num_classes, self.dataset.num_features, counter)
                plt.hist(self.dataset.data[c*self.dataset.per_class:(c+1)*self.dataset.per_class, feature], bins=20, range=[0,8])
                if c == 0: plt.title("Feature: " + str(feature))
                if feature == 0: plt.ylabel("Class: " + str(c))
        self.save_figure()

    def plot_MSE(self):
        self.new_figure()
        plt.plot(self.mean_square_error)
        plt.title("Square Error")
        self.save_figure()

    # helper functions for training
    def _MSEgradient(self, G, T, X):
        return ((G - T) * G * (1 - G)) @ X.T  # eq (22) in Johnsen, rewritten with G, T, and X matrices 
                                              # for significantly shorter comutation time

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))         # eq (20) in Johnsen
