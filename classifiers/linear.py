import numpy as np

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
        self.step_size        = 0.02
        self.max_iterations   = 1000
        self.threshold        = 0.01

    def train(self, training_first = True):
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

        t = np.asarray([np.eye(1, self.num_classes, c).flatten() for c in range(self.num_classes) for _ in range(self.num_train)]).transpose()
        self.W = np.random.uniform(low=-1, high=1, size=(self.num_classes, self.num_features + 1))

        terminating_criteria = False
        iteration = 0
        while not terminating_criteria:
            x = np.hstack((self.trainv, np.ones((self.num_train * self.num_classes, 1)))).transpose() # transformation [x^T 1]^T -> x
            z = self.W @ x
            g = sigmoid(z)                                                                            # eq (20) in Johnson
            gradient = MSEgradient(g, t, x)
            self.W = self.W - self.step_size * gradient                                               # eq (23) in Johnson
            terminating_criteria = iteration > self.max_iterations or np.linalg.norm(gradient) < self.threshold
            iteration += 1
            if terminating_criteria:
                print(gradient)
                print(sigmoid(self.W @ x))
                print(np.linalg.norm(gradient))
                print(iteration)

    def test(self):
        pass

    def print_performance(self):
        print("TRAINING SET")
        print("Confusion Matrix:")
        print(self.train_confusion_matrix)
        print(f"Detection rate: {round(self._get_detection_rate(self.train_confusion_matrix)*100, 2)}%")
        print("\nTEST SET")
        print("Confusion Matrix:")
        print(self.test_confusion_matrix)
        print(f"Detection rate: {round(self._get_detection_rate(self.test_confusion_matrix)*100, 2)}%")
    
    def _get_detection_rate(self, confusion_matrix):
        return np.trace(confusion_matrix)/np.sum(confusion_matrix)

# helper functions for training
def MSEgradient(g, t, x):
        return (g - t) * g * (1 - g) @ np.transpose(x)               # eq (22) in Johnson

def sigmoid(z):
        return 1 / (1 + np.exp(-z))                                  # eq (20) in Johnson
