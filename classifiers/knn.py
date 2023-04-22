import numpy as np
import scipy.spatial
import concurrent.futures
from matplotlib import pyplot as plt

from .classifier import Classifier

class NN(Classifier):
    """
    Classifier based on the nearest neighbor rule.
    """

    def __init__(self, data_path):
        super().__init__()
        data_all = scipy.io.loadmat(data_path)
        self.trainv   = data_all['trainv']             # training data
        self.trainlab = data_all['trainlab'].flatten() # training labels
        self.testv    = data_all['testv']              # test data
        self.testlab  = data_all['testlab'].flatten()  # test labels

        self.num_train = data_all['num_train'][0,0] # number of training samples
        self.num_test  = data_all['num_test'][0,0]  # number of testing samples
        self.row_size  = data_all['row_size'][0,0]  # number of rows in a single image
        self.col_size  = data_all['col_size'][0,0]  # number of columns in a single image
        self.vec_size  = data_all['vec_size'][0,0]  # number of elements in an image vector

        self.confusion_matrix = np.zeros([self.row_size, self.col_size])

    def test(self, num_chunks):
        if self.trainlab.shape[0] % num_chunks != 0:
            raise Exception("num_chunks is not evenly divisible by the number of training samples")
        if  self.testlab.shape[0] % num_chunks != 0:
            raise Exception("num_chunks is not evenly divisible by the number of test samples")

        print("Testing 1NN classifier. This may take a few seconds...")

        # each chunk of test samples can be classified in parallell as the sets are independent
        # therefore, threads are used to test in parallell to speed up the process.
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_chunks) as executor:
            # executor.map(args) returns a list of classified labels for each chunk in order
            # all classified labels are then collected in a single numpy array by concatenating this list of arrays
            self.classified_labels = np.concatenate(list(executor.map(
                self._classify_chunk, 
                np.split(self.trainv, num_chunks), 
                np.split(self.testv, num_chunks),
                np.split(self.trainlab, num_chunks),
            )))
        self.confusion_matrix = scipy.stats.contingency.crosstab(self.testlab, self.classified_labels).count

    def plot_misclassified(self, selection_size):
        misclassified_filter = self.classified_labels != self.testlab.flatten()
        misclassified_labels = self.classified_labels[misclassified_filter] # all misclassified labels
        misclassified_labels = misclassified_labels[:selection_size]        # selection_size first misclassified labels
        image_data = self.testv[misclassified_filter, :]                    # all misclassified data
        image_data = image_data[:selection_size, :]                         # selection_size first misclassified data
        correct_labels = self.testlab.flatten()[misclassified_filter]
        correct_labels = correct_labels[:selection_size]
        self._plot_selection(image_data, misclassified_labels, correct_labels)

    def plot_correctly_classified(self, selection_size):
        correctly_classified_filter = self.classified_labels == self.testlab
        correctly_classified_labels = self.classified_labels[correctly_classified_filter] # all correctly classified labels
        correctly_classified_labels = correctly_classified_labels[:selection_size]        # selection_size first correctly classified labels
        image_data = self.testv[correctly_classified_filter, :]                           # all correctly classified data
        image_data = image_data[:selection_size, :]                                       # selection_size first correctly classified data
        correct_labels = self.testlab[correctly_classified_filter]
        correct_labels = correct_labels[:selection_size]
        self._plot_selection(image_data, correctly_classified_labels, correct_labels)

    def _classify_chunk(self, train_subset_data, test_subset_data, train_subset_labels):
        """
        Classifies a chunk of the test data according to the nearest neighbor rule
        on a subset of the training data.
        """
        dist = scipy.spatial.distance_matrix(train_subset_data, test_subset_data) # calculate the distance matrix
        nearest_neighbor_index_array = np.argmin(dist, axis=0)                    # find the indexes of the nearest neighbors
        return np.take(train_subset_labels, nearest_neighbor_index_array)         # classified label is the label of the nearest neighbor

    def _plot_selection(self, data, classified_labels, correct_labels):
        selection_size = len(data) # must be even
        super().new_figure()
        for index in range(selection_size):
            plt.subplot(2, selection_size//2, index+1)
            square_test_image = np.reshape(data[index], [self.row_size, self.col_size])
            plt.imshow(square_test_image, interpolation='nearest')
            plt.title(f"Classified: {classified_labels[index]}\nActual: {correct_labels[index]}")
