import numpy as np
from matplotlib import pyplot as plt # remove
import scipy.stats.contingency

from .classifier import Classifier

class Linear(Classifier):
    """
    Classifier based on a linear decision rule.
    """

    def __init__(self, data_path):
        super().__init__()
        self.data                   = np.loadtxt(data_path, delimiter=',', usecols=[0,1,2,3])
        self.labels                 = np.loadtxt(data_path, delimiter=',', usecols=[4], dtype='|S15')
        self.num_features           = self.data.shape[1]
        self.num_classes            = len(np.unique(self.labels))
        self.train_confusion_matrix = np.zeros([self.num_classes, self.num_classes])
        train_test_ratio            = 3 / 5
        self.num_train              = int(self.labels.shape[0] / self.num_classes * train_test_ratio)
        self.num_test               = int(self.labels.shape[0] / self.num_classes * (1-train_test_ratio))
        self.per_class              = self.num_train + self.num_test
        self.test_confusion_matrix  = np.zeros([self.num_classes, self.num_classes])

        # training parameters
        self.step_size        = 0.001
        self.max_iterations   = 10000
        self.threshold        = 0.1

    def train(self, training_first = True):
        print("Testing linear classifier...")
        if training_first: # select first subset as training set and last as test set
            self.trainv   = self.data[[row + self.per_class * c for row in range(self.num_train) for c in range(self.num_classes)],:]
            self.trainlab = self.labels[[row + self.per_class * c for row in range(self.num_train) for c in range(self.num_classes)]]
            self.testv    = self.data[[self.num_train + row + self.per_class * c for row in range(self.num_test) for c in range(self.num_classes)],:]
            self.testlab  = self.labels[[self.num_train + row + self.per_class * c for row in range(self.num_test) for c in range(self.num_classes)]]
        else:              # select first subset as test set and last as training set
            self.testv    = self.data[[row + self.per_class * c for row in range(self.num_test) for c in range(self.num_classes)],:]
            self.testlab  = self.labels[[row + self.per_class * c for row in range(self.num_test) for c in range(self.num_classes)]]
            self.trainv   = self.data[[self.num_test + row + self.per_class * c for row in range(self.num_train) for c in range(self.num_classes)],:]
            self.trainlab = self.labels[[self.num_test + row + self.per_class * c for row in range(self.num_train) for c in range(self.num_classes)]]

        t = np.asarray([np.eye(1, self.num_classes, c).flatten()     # t = [ 1 1 ... 0 0 ... 0 0
                        for c in range(self.num_classes)             #       0 0 ... 1 1 ... 0 0
                        for _ in range(self.num_train)]).transpose() #       0 0 ... 0 0 ... 1 1 ]
        x = np.hstack((self.trainv, np.ones((self.num_train * self.num_classes, 1)))).transpose() # transformation [x^T 1]^T -> x
        #self.W = np.random.randn(self.num_classes, self.num_features + 1)
        self.W = np.zeros([self.num_classes, self.num_features + 1])

        mse = []

        terminating_criteria = False
        iteration = 0
        while not terminating_criteria:
            z = self.W @ x
            g = self._sigmoid(z)                               # eq (20) in Johnson
            gradient = self._MSEgradient(g, t, x)
            self.W = self.W - self.step_size * gradient       # eq (23) in Johnson
            iteration += 1
            mse.append(0.5 * np.sum((g - t) * (g - t), axis=1).sum())
            terminating_criteria = iteration > self.max_iterations or mse[iteration - 1] < self.threshold
        print("W @ x0:", g[:,0], "=> classified:", np.argmax(g[:,0]), "| true class:", 0)
        print("W @ x30:", g[:,30], "=> classified:", np.argmax(g[:,30]), "| true class:", 1)
        print("W @ x60:", g[:,60], "=> classified:", np.argmax(g[:,60]), "| true class:", 2)
        print("terminated at iteration:", iteration-1, "with |grad MSE| =", np.linalg.norm(gradient))
        self.train_confusion_matrix = scipy.stats.contingency.crosstab(np.argmax(g, axis=0), np.argmax(t, axis=0)).count
        
        # TESTING
        Classifier.figure_counter += 1
        plt.figure(Classifier.figure_counter)
        plt.plot(mse)
        plt.title("MSE")

    def test(self):
        pass

    def print_performance(self):
        print("\n~~ PERFORMANCE ~~")
        print("TRAINING SET")
        print("Confusion Matrix:")
        print(self.train_confusion_matrix)
        print(f"Detection rate: {round(self._get_detection_rate(self.train_confusion_matrix)*100, 2)}%")
        print("\nTEST SET")
        print("Confusion Matrix:")
        print(self.test_confusion_matrix)
        print(f"Detection rate: {round(self._get_detection_rate(self.test_confusion_matrix)*100, 2)}%")
    
    def plot_histograms(self):
        print(self.data.shape)
        Classifier.figure_counter += 1
        plt.figure(Classifier.figure_counter)
        counter = 0
        for c in range(self.num_classes):
            for feature in range(self.num_features):
                counter += 1
                plt.subplot(self.num_classes, self.num_features, counter)
                plt.hist(self.data[c*self.per_class:(c+1)*self.per_class, feature], bins=20, range=[0,8])
                if c == 0: plt.title("Feature: " + str(feature))
                if feature == 0: plt.ylabel("Class: " + str(c))

    def _get_detection_rate(self, confusion_matrix):
        return np.trace(confusion_matrix) / np.sum(confusion_matrix)

    # helper functions for training
    def _MSEgradient(self, g, t, x):
        return (g - t) * g * (1 - g) @ x.T   # eq (22) in Johnson
        # SIGNIFICANTLY SLOWER ALTERNATIVE:
        # gradient = np.zeros([self.num_classes, self.num_features + 1])
        # for k in range(self.num_train * self.num_classes):
        #     gradient += (g[:,k:k+1] - t[:,k:k+1]) * g[:,k:k+1] * (1 - g[:,k:k+1]) @ x[:,k:k+1].T
        # return gradient

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))          # eq (20) in Johnson
