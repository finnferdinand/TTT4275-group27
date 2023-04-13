import scipy.io

# Import dataset
DATA_ALL_PATH = 'MNist_ttt4275/data_all.mat'
data_all = scipy.io.loadmat(DATA_ALL_PATH)
trainv = data_all['trainv'] # training data
testv = data_all['testv']   # testing data

num_train = data_all['num_train']
num_test = data_all['num_test']
row_size = data_all['row_size']
col_size = data_all['col_size']
vec_size = data_all['vec_size']

# Iterating through chunks of CHUNK_SIZE images from the training set
CHUNK_SIZE = 1000
n_chunks = num_train // CHUNK_SIZE
print(n_chunks)
for chunk in range(n_chunks):
    chunk_data = trainv[CHUNK_SIZE*chunk : CHUNK_SIZE*(chunk+1), vec_size]
    # print(chunk_data.shape)
