### Import required packages ###
import numpy as np

###############################################################################
# This script combines and weights arrays of neighborhoood graphs. It finally
# produces an array of unique neighborhood graphs used for training the
# autoencoder
###############################################################################

### Load all non-normalized, non-weighted, NGA signature arrays ###
df = []
df.append(np.load("Results/Neigbhorhood_Graphs/Multi-Flavored_500_particles/cry_0.npy"))
df.append(np.load("Results/Neigbhorhood_Graphs/Multi-Flavored_500_particles/cry_1.npy"))
df.append(np.load("Results/Neigbhorhood_Graphs/Multi-Flavored_500_particles/cry_2.npy"))
df.append(np.load("Results/Neigbhorhood_Graphs/Multi-Flavored_500_particles/cry_3.npy"))
df.append(np.load("Results/Neigbhorhood_Graphs/Multi-Flavored_500_particles/cry_4.npy"))
df.append(np.load("Results/Neigbhorhood_Graphs/Multi-Flavored_500_particles/cry_5.npy"))
df.append(np.load("Results/Neigbhorhood_Graphs/Multi-Flavored_500_particles/cry_6.npy"))
df.append(np.load("Results/Neigbhorhood_Graphs/Multi-Flavored_500_particles/cry_7.npy"))
df.append(np.load("Results/Neigbhorhood_Graphs/Multi-Flavored_500_particles/cry_8.npy"))
df.append(np.load("Results/Neigbhorhood_Graphs/Multi-Flavored_500_particles/cry_9.npy"))
df.append(np.load("Results/Neigbhorhood_Graphs/Multi-Flavored_500_particles/cry_10.npy"))

### Stack list into array ###
df_array = np.vstack(df)

### Reduce array down to unique NGA signatures only ###
df_array_unique = np.unique(df_array, axis = 0)
        
### Create weight vector from relative orbit importances ###
o = np.array([1, 2, 2, 2, 3, 4, 3, 3, 4, 3, 4, 4, 4, 4, 3, 4, 6, 
              5, 4, 5, 6, 6, 4, 4, 4, 5, 7, 4, 6, 6, 7, 4, 6, 6,
              6, 5, 6, 7, 7, 5,7, 6, 7, 6, 5, 5, 6, 8, 7, 6, 6, 
              8, 6, 9, 5, 6, 4, 6, 6, 7, 8, 6, 6, 8, 7, 6, 7, 7, 
              8, 5, 6, 6, 4],dtype=np.float)

w = 1. - o / 73.

### Create and save weighted array of unique neighborhood graphs ###
# Note that each array is normalized such that its sum is equal to 1
weighted_list = []
for i in range(0, len(df_array_unique)):
    weighted_list.append(df_array_unique[i,:]*w/np.sum(df_array_unique[i,:]*w))
    
weighted_array = np.vstack(weighted_list) 

np.save('Results/Neigbhorhood_Graphs/Multi-Flavored_500_particles/weighted_array.npy', weighted_array)