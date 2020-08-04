### Import required packages
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

###############################################################################
#This script employs agglomerative hierarchical clustering (using Ward's method)
#to partition the low-dimensional space created by the trained encoder model.
# Note that only the low-dimensional representations of the UNIQUE neighborhood
# graphs are clustered.
###############################################################################


### Enter relevant paths
input_path = "Results/Dimensionality_Reduction/" # where low-dimensional data
# is stored

path = "Results/Clustering/" # path where clustering results are stored

### Load low-dimensional representations of neighborhood graphs ###
lowd_train = np.load(input_path+"LowD_Train.npy")

### Implement agglomerative hierarhical clustering. Note that Ward's linkage
# is used although, other linkage methods are available
nc = 20 # choose number of clusters
model = AgglomerativeClustering(n_clusters=nc, linkage='ward') # initialize
# model
yhat = model.fit_predict(lowd_train) # get prediction results
np.save(path+"yhat_"+str(nc)+"clusters.npy", yhat) # save results

# Note that the array "yhat" produces the identity of the cluster that each
# data point begins to. This identity is a number from 0 to the total number
# of clusters

### Plot cluster tree ###
# Calculate linkage only. Note that the linkage is calculated implicitly
# in the lines above, but the linkage needs to be calculated explicitly here
# to complete plot
linked = linkage(lowd_train, 'ward')

# "Dendrogram" is the official word for cluster tree. The following function
# has a lot of options associated with it. We provide the simplest option for
# plotting a cluster tree below. The x-axis shows the number of points that
# belong to each leaf in the tree
plt.figure(figsize=(16,9))
dendrogram(
            linked,
            truncate_mode='lastp',  # show only the last "p" merged clusters
            p=nc, color_threshold=20)

plt.xlabel('Number of Points in Cluster', fontsize=12)
plt.ylabel('Ward\'s Distance', fontsize=12)
plt.savefig(path+'Dendrogram_'+str(nc)+'_clusters.png')
