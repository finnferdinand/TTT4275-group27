import numpy as np
from matplotlib import pyplot as plt # remove
import scipy.stats.contingency

from .classifier import Classifier

class Linear(Classifier):
    """
    Classifier based on a linear decision rule, using gradient descent
    training with minimum mean square error cost function.
    It uses the method described in Johnson (2017) pp. 9-10 & 15-18.

    Equations are implemented as described in Johnson, however, in
    order to speed up computation time and for shorter notation
    the vectors g_k, t_k, and x_k have been converted into the matrices
    g, t, and x containing the values for all k. This avoids using
    sums and for loops many places, which are computationally expensive.
    """

    def __init__(self, data_path):
        super().__init__()
        self.data                   = np.loadtxt(data_path, delimiter=',', usecols=[0,1,2,3])
        self.num_data_points        = self.data.shape[0]
        self.num_features           = self.data.shape[1]
        labels                      = np.loadtxt(data_path, delimiter=',', usecols=[4], dtype='|S15')
        self.num_classes            = len(np.unique(labels))
        self.train_confusion_matrix = np.zeros([self.num_classes, self.num_classes])
        train_test_ratio            = 3 / 5
        self.num_train              = int(self.num_data_points / self.num_classes * train_test_ratio)
        self.num_test               = int(self.num_data_points / self.num_classes * (1-train_test_ratio))
        self.per_class              = self.num_train + self.num_test
        self.test_confusion_matrix  = np.zeros([self.num_classes, self.num_classes])
        self.mse                    = []

        self.step_size              = 0.005  # alpha as in eq (23) in Johnson
        self.max_iterations         = 1000   # maximum number of iterations
        self.threshold              = 0.5    # terminate training if ||grad MSE|| < this number

    def train(self, training_first = True):
        print("Training linear classifier...")
        if training_first: # select first subset as training set and last as test set
            self.trainv = self.data[[row % self.per_class < self.num_train for row in range(self.num_data_points)],:]
            self.testv  = self.data[[row % self.per_class >= self.num_train for row in range(self.num_data_points)],:]
        else:              # select first subset as test set and last as training set
            self.testv  = self.data[[row % self.per_class < self.num_test for row in range(self.num_data_points)],:]
            self.trainv = self.data[[row % self.per_class >= self.num_test for row in range(self.num_data_points)],:]

        t = np.asarray([np.eye(1, self.num_classes, c).flatten()     # t = [ 1 1 ... 0 0 ... 0 0
                        for c in range(self.num_classes)             #       0 0 ... 1 1 ... 0 0
                        for _ in range(self.num_train)]).transpose() #       0 0 ... 0 0 ... 1 1 ]
        x = np.hstack((self.trainv, 
                       np.ones((self.num_train * self.num_classes, 1))
                       )).transpose()                                # transformation [x^T 1]^T -> x
        self.W = np.zeros([self.num_classes, self.num_features + 1]) # initialize as (C,D+1) zero matrix

        terminating_criteria = False
        iteration = 0
        while not terminating_criteria:
            z = self.W @ x
            g = self._sigmoid(z)                                # eq (20) in Johnson
            gradient = self._MSEgradient(g, t, x)
            self.W = self.W - self.step_size * gradient         # eq (23) in Johnson
            iteration += 1
            mse = 0.5 * np.sum((g - t) * (g - t), axis=1).sum() # eq (19) in Johnson
            self.mse.append(mse)
            terminating_criteria = iteration > self.max_iterations \
                                 or np.linalg.norm(gradient) < self.threshold

        print(f"Training terminated at iteration: {iteration-1}, with ||grad MSE|| = {round(np.linalg.norm(gradient),2)}")
        self.train_confusion_matrix = scipy.stats.contingency.crosstab(np.argmax(g, axis=0), np.argmax(t, axis=0)).count

    def test(self):
        print("Testing linear classifier...")
        t = np.asarray([np.eye(1, self.num_classes, c).flatten()    # t = [ 1 1 ... 0 0 ... 0 0
                        for c in range(self.num_classes)            #       0 0 ... 1 1 ... 0 0
                        for _ in range(self.num_test)]).transpose() #       0 0 ... 0 0 ... 1 1 ]
        x = np.hstack((self.testv, 
                       np.ones((self.num_test * self.num_classes, 1))
                       )).transpose()                               # transformation [x^T 1]^T -> x
        z = self.W @ x
        g = self._sigmoid(z)
        self.test_confusion_matrix = scipy.stats.contingency.crosstab(np.argmax(g, axis=0), np.argmax(t, axis=0)).count

    def print_performance(self):
        print("\n~~ PERFORMANCE ~~")
        print("Training Set CM" + self.num_classes * "   " + "  " + "Test Set CM")
        for c in range(self.num_classes):
            print(f"{self.train_confusion_matrix[c,:]}" + "   " * self.num_classes + "       " + f"{self.test_confusion_matrix[c,:]}")
        print(f"Det. rate: {round(self._get_detection_rate(self.train_confusion_matrix)*100, 2)}%" +
              f"   " * self.num_classes +
              f"Det. rate: {round(self._get_detection_rate(self.test_confusion_matrix)*100, 2)}%")
    
    def plot_histograms(self):
        print("Producing histograms...")
        super().new_figure()
        counter = 0
        for c in range(self.num_classes):
            for feature in range(self.num_features):
                counter += 1
                plt.subplot(self.num_classes, self.num_features, counter)
                plt.hist(self.data[c*self.per_class:(c+1)*self.per_class, feature], bins=20, range=[0,8])
                if c == 0: plt.title("Feature: " + str(feature))
                if feature == 0: plt.ylabel("Class: " + str(c))

    def plot_MSE(self):
        super().new_figure()
        plt.plot(self.mse)
        plt.title("MSE")

    def _get_detection_rate(self, confusion_matrix):
        return np.trace(confusion_matrix) / np.sum(confusion_matrix)

    # helper functions for training
    def _MSEgradient(self, g, t, x):
        return (g - t) * g * (1 - g) @ x.T   # eq (22) in Johnson, rewritten with g, t, and x matrices 
                                             # for significantly shorter comutation time

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))          # eq (20) in Johnson
