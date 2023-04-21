import scipy.io
from matplotlib import pyplot as plt
import numpy as np

from classifiers.knn import NN
from classifiers.linear import Linear

np.set_printoptions(suppress=True)
np.random.seed(1) # set seed for debugging purposes

iris_data_path = 'data/Iris_TTT4275/iris.data'
digits_data_path = 'data/MNist_ttt4275/data_all.mat'

if __name__ == "__main__":
    print("\n\n----- IRIS -----")    
    print("Using first 30 as training set and last 20 as testing set")    
    linear_classifier = Linear(iris_data_path)
    linear_classifier.train()
    linear_classifier.test()
    linear_classifier.print_performance()

    exit(0) # skipping working on linear classifier

    print("Using first 20 as training set and last 30 as testing set")    
    linear_classifier = Linear(iris_data_path)
    linear_classifier.train(training_first=False)
    linear_classifier.test()
    linear_classifier.print_performance()

    print("\n\n---- DIGITS ----")
    print("Using full traning set as templates")
    nn_classifier = NN(digits_data_path)
    nn_classifier.test(num_chunks=50)
    nn_classifier.print_performance()
    # plot a selection of misclassified and correctly classified test samples
    selection_size = 10
    nn_classifier.plot_misclassified(selection_size)
    nn_classifier.plot_correctly_classified(selection_size)

    # TODO: KNN classifier


    plt.show()
