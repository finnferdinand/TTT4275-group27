#!/usr/bin/env python

from matplotlib import pyplot as plt

from experiments.digits import digits
from experiments.iris import iris

__author__ = "Finn Ferdinand Sandvand and Christian Le"
__copyright__ = "Copyright 2023"
__license__ = "MIT"

if __name__ == "__main__":
    iris()
    digits()

    plt.show()
