import numpy as np
from matplotlib import pyplot as plt

class Classifier(object):
    """
    An abstract class for a general classifier.
    """
    figure_counter = 0

    def __init__(self):
        self.confusion_matrix = None
        self.classified_labels = None

    def print_performance(self):
        print("Confusion Matrix:")
        print(self.confusion_matrix)
        print(f"Detection rate: {round(self._get_detection_rate()*100, 2)}%")

    def _get_detection_rate(self):
        return np.trace(self.confusion_matrix)/np.sum(self.confusion_matrix)

    def new_figure(self):
        Classifier.figure_counter += 1
        plt.figure(Classifier.figure_counter)
