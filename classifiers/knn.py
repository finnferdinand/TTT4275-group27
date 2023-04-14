import numpy as np
import scipy.io
import scipy.spatial

# IMPORT DATASET
MODULE_NAME = '.'.join(__name__.split('.')[:-1])
DATA_ALL_PATH = MODULE_NAME + '/MNist_ttt4275/data_all.mat'
data_all = scipy.io.loadmat(DATA_ALL_PATH)

trainv   = data_all['trainv']   # training data
trainlab = data_all['trainlab'] # training labels
testv    = data_all['testv']    # test data
testlab  = data_all['testlab']  # test labels

num_train = data_all['num_train'][0,0] # number of training samples
num_test  = data_all['num_test'][0,0]  # number of testing samples
row_size  = data_all['row_size'][0,0]  # number of rows in a single image
col_size  = data_all['col_size'][0,0]  # number of columns in a single image
vec_size  = data_all['vec_size'][0,0]  # number of elements in an image vector

def classifier_1nn():
    print("Testing 1NN classifier. This may take a few seconds...")

    confusion_matrix = np.zeros([10, 10]) # row = true class, col = classified class
    SELECTION_SIZE = 10
    misclassified_counter = 0
    misclassified_selection = np.zeros([SELECTION_SIZE, vec_size])
    misclassified_labels = []
    correctly_classified_counter = 0
    correctly_classified_selection = np.zeros([SELECTION_SIZE, vec_size])
    correctly_classified_labels = []

    # Iterating through chunks of CHUNK_SIZE images from the training set
    TRAINING_CHUNK_SIZE = 1000                    # full training set only for preliminary testing (TODO: Set to 1000)
    NUM_CHUNKS = num_train // TRAINING_CHUNK_SIZE # number of training samples in a single chunk
    TEST_CHUNK_SIZE = num_test // NUM_CHUNKS
    for chunk in range(NUM_CHUNKS):
        print(f"testing progress: {round(chunk/NUM_CHUNKS*100)}%", end="\r")
        training_chunk = trainv[TRAINING_CHUNK_SIZE*chunk:TRAINING_CHUNK_SIZE*(chunk+1), 0:vec_size] # chunk slice from full training set
        test_chunk = testv[TEST_CHUNK_SIZE*chunk:TEST_CHUNK_SIZE*(chunk+1), 0:vec_size]
        dist = scipy.spatial.distance_matrix(training_chunk, test_chunk)                             # distance vector from test image to chunk
        for test_sample in range(TEST_CHUNK_SIZE):
            nn_index = np.argmin(dist[0:TRAINING_CHUNK_SIZE,test_sample])
            classified_label = trainlab[nn_index+chunk*TRAINING_CHUNK_SIZE,0]
            actual_label = testlab[test_sample+chunk*TEST_CHUNK_SIZE,0]
            confusion_matrix[actual_label, classified_label] += 1
            if misclassified_counter < SELECTION_SIZE and classified_label != actual_label:
                misclassified_selection[misclassified_counter] = testv[test_sample+chunk*TEST_CHUNK_SIZE, 0:vec_size]
                misclassified_counter += 1
                misclassified_labels.append(classified_label)
            elif correctly_classified_counter < SELECTION_SIZE and classified_label == actual_label:
                correctly_classified_selection[correctly_classified_counter] = testv[test_sample+chunk*TEST_CHUNK_SIZE, 0:vec_size]
                correctly_classified_counter += 1
                correctly_classified_labels.append(classified_label)
        
        #if chunk > 5: break # ONLY FOR TESTING 
    
    misclassified = (misclassified_selection, misclassified_labels)
    correctly_classified = (correctly_classified_selection, correctly_classified_labels)
    return (confusion_matrix, misclassified, correctly_classified)
