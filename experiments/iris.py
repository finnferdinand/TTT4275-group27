#!/usr/bin/env python

from classifiers.linear import Linear
from datasets.irisdataset import IrisDataSet

__author__ = "Finn Ferdinand Sandvand and Christian Le"
__copyright__ = "Copyright 2023"
__license__ = "MIT"

iris_data_path = 'datasets/data/Iris_TTT4275/iris.data'

def iris():
    # Task 1c) (First 30 as training, last 20 as testing)
    print("---------------------------- IRIS ----------------------------")  
    print(".: Experiment A.1 :.")    
    linear_classifier = Linear(
        IrisDataSet(data_path = iris_data_path, train_test_ratio = 3 / 5), 
        Linear.Configuration(step_size = 0.007, max_iterations = 1500, threshold = 0.3),
    )
    linear_classifier.log_write("---------------------------- IRIS ----------------------------\n")  
    linear_classifier.train(training_first = True)
    linear_classifier.test()
    linear_classifier.log_performance(".: Experiment A.1 :.")
    linear_classifier.plot_MSE()

    # Inspecting dataset
    linear_classifier.plot_histograms()

    # Task 1d) (Last 30 as training, first 20 as testing)
    print("\n.: Experiment A.2 :.")    
    linear_classifier.train(training_first = False)
    linear_classifier.test()
    linear_classifier.log_performance(".: Experiment A.2 :.")

    # Task 2) (Remove features, one by one)
    print("\n.: Experiment A.3 :.")    
    linear_classifier.train(training_first = True, selected_features=[0,2,3])
    linear_classifier.test()
    linear_classifier.log_performance(".: Experiment A.3 :.")

    print("\n.: Experiment A.4 :.")    
    linear_classifier.train(training_first = True, selected_features=[2,3])
    linear_classifier.test()
    linear_classifier.log_performance(".: Experiment A.4 :.")

    print("\n.: Experiment A.5 :.")    
    linear_classifier.train(training_first = True, selected_features=[3])
    linear_classifier.test()
    linear_classifier.log_performance(".: Experiment A.5 :.")
