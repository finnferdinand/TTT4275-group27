import scipy.io
from matplotlib import pyplot as plt
import numpy as np

from classifiers.knn import NN
from classifiers.linear import Linear

np.set_printoptions(suppress=True)


print("\n\n----- IRIS -----")
# linear classifier
iris_data_path = 'Iris_TTT4275/iris.data'
linear_classifier = Linear(iris_data_path)
linear_classifier.train()
linear_classifier.test()

# performance
linear_classifier.print_performance()


print("\n\n---- DIGITS ----")
# 1NN classifier
digits_data_path = 'MNist_ttt4275/data_all.mat'
nn_classifier = NN(digits_data_path)
nn_classifier.test(50)

# performance
nn_classifier.print_performance()

# plot a selection of misclassified and correctly classified test samples
selection_size = 10
nn_classifier.plot_misclassified(selection_size)
nn_classifier.plot_correctly_classified(selection_size)

# TODO: KNN classifier


plt.show()
