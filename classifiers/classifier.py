#!/usr/bin/env python
"""
An abstract class for a general classifier.
"""

import numpy as np
from matplotlib import pyplot as plt

__author__ = "Finn Ferdinand Sandvand and Christian Le"
__copyright__ = "Copyright 2023"
__license__ = "MIT"

class Classifier(object):
    figure_counter = 0

    def __init__(self):
        np.set_printoptions(suppress = True)
        self.confusion_matrix = None
        self.classified_labels = None

    def print_performance(self):
        print("Confusion Matrix:")
        print(self.confusion_matrix)
        print(f"Detection rate: {round(self.get_detection_rate(self.confusion_matrix)*100, 2)}%")

    def get_detection_rate(self, confusion_matrix):
        return np.trace(confusion_matrix) / np.sum(confusion_matrix)

    def new_figure(self):
        Classifier.figure_counter += 1
        plt.figure(Classifier.figure_counter)
