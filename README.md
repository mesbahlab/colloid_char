# colloid_char
The provided scripts implement a three-step framework for colloidal self-assembly state characterization as described in "Deep learning for characterizing the time evolution of three-dimensional colloidal self-assembly systems" (O'Leary, et al., 2020) The first step establishes neighborhood graphs with a precise methodology that has been shown to be robust to thermal fluctuations and capable of describing complex topologies. The second step uses deep learning techniques to reduce the dimensionality of the neighborhood graphs. The third step employs agglomerative hierarchical clustering to partition the low-dimensional space and assign physically meaningful classifications to the resulting partitions.Each of the provided scripts is described briefly below. Please feel free to email any and all questions about using these scripts to jared.oleary@berkeley.edu

# Read_GDV.py 
The package provided by "" will output neigbhorhood graphs from XYZ files that describe particle positions. These neighborhood graphs are output in terms of GDV files. This script extracts the neighborhood graphs from these files.

# Combine_GDV.py
This script combines neighborhood graphs extracted from different GDV files and records the unique neighborhood graphs among them only. The script then weighs and normalizes each of the neighborhood graphs.

# Train.py
This script trains an autoencoder based on the unique neighborhood graphs provided by Combined_GDV.py. Various hyper-parameter choices are explained within the script itslef

# Cluster.py
This script partitions any low-dimensional space using agglomerative hierarhical clustering. A default of 20 clusters was chosen but can be changed.

# Test.py
This script uses the results of the previous scripts to characterize colloidal SA states that were not used to train the autoencoder (i.e., neighborhood graphs from an independent data set) This script weighs, normalizes, and scales the neighborhood graphs and then reduces their dimensionality with the autoencoder that was trained in Train.py. It then compares the positions of the low-dimensional representations of the data used to train the autoencoder with the low-dimensional positions of the independent data set and classifies the positions accordingly. For example, if the low-dimensional representation of a neighborhood graph from the independent data set is [15.17, 3.508163, 18.233894] and the closest low-dimensional point in the training data set adopts class C10, then the particle from the independent data set adopts this class as well

# Provided Data
All provided scripts have been run one time using the training and testing data from "Deep learning for characterizing the time evolution of three-dimensional colloidal self-assembly systems" (O'Leary, et al., 2020). This data (and results garnered from this data) are provided within the "Trajectory_Data" and "Results" folders. The user should "unzip" all zip files before running any of the scripts and make sure that all paths specified in the scripts of interest align with their corresponding data/results folders.

Note that several subfolders are titled "Paper Results." These include the relevant results presented in the above paper. The "Paper Results" are slightly different due to some unavoidable run-to-run variation in the autoencoder training. Suggestions to decrease this variation are included within Train.py (e.g., train for more epochs).


