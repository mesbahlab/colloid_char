### Import required packages
import numpy as np
from scipy.spatial import KDTree
from keras.models import model_from_json

###############################################################################
# The purpose of this script is to characterize data that was NOT used to train
# the autoencoder. The encoder is first used to reduce the dimensionality
# of a matrix of neighborhoood graphs. The script next identifies the training
# data low-dimensional points that are closest to teh low-dimensional
# representations of the incoming data and classified particles accordingly.
# For example, if the low-dimensional representation of an incoming data point
# is [15.17,  3.508163, 18.233894] and the closest training data point to this
# to this point is classified as C2, then the incoming data point adopts this
# class
###############################################################################

### Load neighborhood graphs of testing data ###
testing_data_raw = np.load("/home/mesbahlab/Documents/Jared/MFUC_3D/Github/Results/Neigbhorhood_Graphs/Multi-Flavored_1000_particles/cry.npy")

### Create weight vector from relative orbit importances ###
o = np.array([1, 2, 2, 2, 3, 4, 3, 3, 4, 3, 4, 4, 4, 4, 3, 4, 6, 
              5, 4, 5, 6, 6, 4, 4, 4, 5, 7, 4, 6, 6, 7, 4, 6, 6,
              6, 5, 6, 7, 7, 5,7, 6, 7, 6, 5, 5, 6, 8, 7, 6, 6, 
              8, 6, 9, 5, 6, 4, 6, 6, 7, 8, 6, 6, 8, 7, 6, 7, 7, 
              8, 5, 6, 6, 4],dtype=np.float)

w = 1. - o / 73.

### Create and save weighted array of unique neighborhood graphs ###
# Note that each array is normalized such that its sum is equal to 1
testing_data_weighted = []
for i in range(0, len(testing_data_raw)):
    testing_data_weighted.append(testing_data_raw[i,:]*w/np.sum(testing_data_raw[i,:]*w))
    
testing_data_weighted = np.vstack(testing_data_weighted) 

### Scale weighted array ###
# Load min and max arrays
min_array = np.load("Results/Dimensionality_Reduction/min_array.npy")
max_array = np.load("Results/Dimensionality_Reduction/max_array.npy")

testing_data_scaled = np.zeros((np.shape(testing_data_raw)))
for i in range(0, len(testing_data_scaled)):
    testing_data_scaled[i,:] = 2*(testing_data_weighted[i,:]-min_array)/(max_array-min_array)-1

### Load encoder ###
# Load json and create model
json_file_1 = open("Results/Dimensionality_Reduction/Encoder.json", 'r')
loaded_model_json_1 = json_file_1.read()
json_file_1.close()
loaded_encoder = model_from_json(loaded_model_json_1)

# Load weights into new model
loaded_encoder.load_weights("Results/Dimensionality_Reduction/Encoder.h5")

### Reduce dimensionality of testing data ###
testing_lowd = loaded_encoder.predict(testing_data_scaled)

### Load low-dimensional training data ###
original_data = np.load("Results/Dimensionality_Reduction/LowD_Train.npy")

### Load cluster labels for low-dimensional training data ###
cluster_labels_original = np.load("Results/Clustering/yhat_20clusters.npy")

### Find cluster labels for testing data ###
# Note that the final product will be a vector of cluster labels whose indices
# exactly correspond to those of the loaded testing data
kdtree = KDTree(original_data)

cluster_id_test_list = []
for i in range(0, len(testing_lowd)):
    sample = testing_lowd[i,:]
    dist,points=kdtree.query(sample,1)
    cluster_id_original = cluster_labels_original[points]
    cluster_id_test_list.append(int(cluster_id_original))

cluster_id_test_array = np.asarray(cluster_id_test_list)

np.save('Results/Testing/cluster_id_test.npy', cluster_id_test_array)

