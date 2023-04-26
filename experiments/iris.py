#!/usr/bin/env python

from classifiers.linear import Linear
from datasets.irisdataset import IrisDataSet

__author__ = "Finn Ferdinand Sandvand and Christian Le"
__copyright__ = "Copyright 2023"
__license__ = "MIT"

iris_data_path = 'datasets/data/Iris_TTT4275/iris.data'

def iris():
    print("---------------------------- IRIS ----------------------------")  
    print(".: Using first 30 as training set and last 20 as testing set :.")    
    linear_classifier = Linear(
        IrisDataSet(data_path = iris_data_path, train_test_ratio = 3 / 5), 
        Linear.Configuration(step_size = 0.007, max_iterations = 1500, threshold = 0.3),
    )
    linear_classifier.train(training_first = True)
    linear_classifier.test()
    linear_classifier.print_performance()
    linear_classifier.plot_MSE()

    # Inspecting dataset
    linear_classifier.plot_histograms()

    print("\n.: Using last 30 as training set and first 20 as testing set :.")    
    linear_classifier.train(training_first = False)
    linear_classifier.test()
    linear_classifier.print_performance()

    print("\n.: Removed most overlapping feature :.")    
    linear_classifier.train(training_first = True, selected_features=[0,2,3])
    linear_classifier.test()
    linear_classifier.print_performance()

    print("\n.: Removed 2 most overlapping features :.")    
    linear_classifier.train(training_first = True, selected_features=[2,3])
    linear_classifier.test()
    linear_classifier.print_performance()

    print("\n.: Removed 3 most overlapping features :.")    
    linear_classifier.train(training_first = True, selected_features=[3])
    linear_classifier.test()
    linear_classifier.print_performance()
