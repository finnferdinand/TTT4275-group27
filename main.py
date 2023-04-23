from matplotlib import pyplot as plt

from classifiers.knn import NN
from classifiers.linear import Linear
from datasets.irisdataset import IrisDataSet 
from datasets.mnistdataset import MNISTDataSet 

iris_data_path = 'datasets/data/Iris_TTT4275/iris.data'
digits_data_path = 'datasets/data/MNist_ttt4275/data_all.mat'

if __name__ == "__main__":
    print("------------------------- IRIS -------------------------")    
    print("Using first 30 as training set and last 20 as testing set")    
    linear_classifier = Linear(
        IrisDataSet(data_path = iris_data_path, train_test_ratio = 3 / 5), 
        Linear.Configuration(step_size = 0.005, max_iterations = 1000, threshold = 0.5),
    )
    linear_classifier.train(training_first=True)
    linear_classifier.test()
    linear_classifier.print_performance()

    print("\nUsing last 30 as training set and first 20 as testing set")    
    linear_classifier.train(training_first=False)
    linear_classifier.test()
    linear_classifier.print_performance()

    # linear_classifier.plot_histograms()

    # print("\n\n------------------------ DIGITS ------------------------")
    # print("Using full traning set as templates")
    # nn_classifier = NN(MNISTDataSet(digits_data_path))
    # nn_classifier.test(num_chunks=50)
    # nn_classifier.print_performance()
    # # plot a selection of misclassified and correctly classified test samples
    # selection_size = 10
    # nn_classifier.plot_misclassified(selection_size)
    # nn_classifier.plot_correctly_classified(selection_size)

    # # TODO: KNN classifier


    plt.show()
