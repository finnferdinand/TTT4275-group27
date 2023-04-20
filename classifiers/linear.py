import numpy as np

from .classifier import Classifier

class Linear(Classifier):
    """
    Classifier based on a linear decision rule.
    """

    def __init__(self, data_path):
        super().__init__()
        self.data             = np.loadtxt(data_path, delimiter=',', usecols=[0,1,2,3])
        self.labels           = np.loadtxt(data_path, delimiter=',', usecols=[4], dtype='|S15')
        self.num_classes      = len(np.unique(self.labels))
        self.confusion_matrix = np.zeros([self.num_classes, self.num_classes])
        train_test_ratio = 3/5
        self.num_train = int(self.labels.shape[0] / self.num_classes * train_test_ratio)
        self.num_test  = int(self.labels.shape[0] / self.num_classes * (1-train_test_ratio))

    def train(self, training_first = True):
        if training_first:
            self.trainv   = self.data[:self.num_train]
            self.trainlab = self.labels[:self.num_train]
            self.testv    = self.data[self.num_train:]
            self.testlab  = self.labels[self.num_train:]
        else:
            self.testv    = self.data[:self.num_test]
            self.testlab  = self.labels[:self.num_test]
            self.trainv   = self.data[self.num_test:]
            self.trainlab = self.labels[self.num_test:]

    def test(self):
        pass
