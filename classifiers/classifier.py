#!/usr/bin/env python
"""
An abstract class for a general classifier.
"""

import numpy as np
from matplotlib import pyplot as plt

from utilities.logger import Logger

__author__ = "Finn Ferdinand Sandvand and Christian Le"
__copyright__ = "Copyright 2023"
__license__ = "MIT"

class Classifier(object):
    figure_counter = 0

    def __init__(self):
        np.set_printoptions(suppress = True)
        self.confusion_matrix = None
        self.classified_labels = None
        self.logger = Logger()

    def log_performance(self, title):
        print("Logging performance...")
        self.logger.write("\n" + title + "\n")
        self.logger.write("Confusion Matrix:\n")
        self.logger.write(f"{self.confusion_matrix}\n")
        self.logger.write(f"Detection rate: {round(self.get_detection_rate(self.confusion_matrix)*100, 2)}%\n\n")

    def get_detection_rate(self, confusion_matrix):
        return np.trace(confusion_matrix) / np.sum(confusion_matrix)

    def new_figure(self):
        Classifier.figure_counter += 1
        plt.figure(Classifier.figure_counter)

    def log_write(self, string):
        self.logger.write(string)
