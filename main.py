import numpy as np
import scipy.io
import scipy.spatial
from matplotlib import pyplot as plt

# IMPORT DATASET
DATA_ALL_PATH = 'MNist_ttt4275/data_all.mat'
data_all = scipy.io.loadmat(DATA_ALL_PATH)

trainv   = data_all['trainv']   # training data
trainlab = data_all['trainlab'] # training labels
testv    = data_all['testv']    # test data
testlab  = data_all['testlab']  # test labels

num_train = data_all['num_train'][0][0] # number of training samples
num_test  = data_all['num_test'][0][0]  # number of testing samples
row_size  = data_all['row_size'][0][0]  # number of rows in a single image
col_size  = data_all['col_size'][0][0]  # number of columns in a single image
vec_size  = data_all['vec_size'][0][0]  # number of elements in an image vector

# CLASSIFY AN IMAGE
test_index = 0 # only testing first image for now, preliminary testing
test_image = testv[test_index:test_index+1, 0:vec_size] # select first image from test set

# Iterating through chunks of CHUNK_SIZE images from the training set
CHUNK_SIZE = num_train             # full training set only for preliminary testing (TODO: Set to 1000)
n_chunks = num_train // CHUNK_SIZE # number of training samples in a single chunk
for chunk in range(n_chunks):
    chunk_data = trainv[CHUNK_SIZE*chunk : CHUNK_SIZE*(chunk+1), 0:vec_size] # chunk slice from full training set
    dist = scipy.spatial.distance_matrix(chunk_data, test_image)             # distance vector from test image to chunk
    nn_index = np.argmin(dist)                                               # find nearest neigbor -> smallest distance to test sample
    print(f"Closest template {nn_index} " 
          f"labeled {trainlab[nn_index+chunk*CHUNK_SIZE][0]} "
          f"with a distance of {round(dist[nn_index][0], 2)}")
    print("Actual label", testlab[test_index][0])

# DRAWING IMAGES
plt.subplot(1,2,1) # test sample image
square_test_image = np.reshape(test_image, [row_size, col_size])
plt.imshow(square_test_image, interpolation='nearest')
plt.title("Test Sample")
plt.subplot(1,2,2) # nearest neigbor image
square_closest_neighbor = np.reshape(chunk_data[nn_index:nn_index+1, 0:vec_size], [row_size, col_size])
plt.imshow(square_closest_neighbor, interpolation='nearest')
plt.title("Nearest Neighbor")
plt.show()
