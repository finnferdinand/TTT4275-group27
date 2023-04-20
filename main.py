import classifiers.knn as knn
import scipy.io
from matplotlib import pyplot as plt
import numpy as np
np.set_printoptions(suppress=True)

## DIGITS
# import
DATA_ALL_PATH = 'MNist_ttt4275/data_all.mat'
data_all = scipy.io.loadmat(DATA_ALL_PATH)

# 1NN classifier
NN = knn.NN(data_all)
NN.test(50)
print("1NN Confusion Matrix:")
print(NN.confusion_matrix)
print(f"Detection rate: {round(NN.get_detection_rate()*100, 2)}%")

selection_size = 10
NN.plot_misclassified(selection_size)
NN.plot_correctly_classified(selection_size)



plt.show()
