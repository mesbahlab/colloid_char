### Import required packages
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras import Sequential
from keras.layers import Dropout
from keras.constraints import maxnorm

###############################################################################
# This script takes a matrix of unique neighborhood graphs, scales the data
# and then trains an autoencocer. Various autoencoder architecture choices are
# indicated throughout the script.
###############################################################################

### Load training data ###
# The training data takes the form of a matrix of all unique neighborhood
# graphs. The matrix dimensions are (Ny, Nx), where Ny is the number of
# unique neighborhood graphs and Nx is the (flattened) dimension of the
# neighborhood graphs.
x_train_raw = (np.load('Results/Neigbhorhood_Graphs/Multi-Flavored_500_particles/weighted_array.npy'))

### Choose relevant folders ###
path = "Results/Dimensionality_Reduction/" # path to save models in 

### Enter relevant parameters for constructing autoencoder network
n_hidden_layers = 2 # number of hidden layers
n_nodes = 1000 # number of nodes per hidden layer
bottleneck_size = 3 # size of bottleneck layer
n_epoch = 2*10**4 # number of epochs used in training. Note more epochs lead
# to more stable results. 10**5 epochs is recommended
activation_function = "tanh" # choice of activation function. "relu" should
# generally be the first choice, however
use_dropout = True # Choose whether or not to use dropout regularization
drop_prob = 0.20 # probability of keeping a given node during dropout
kc_max_norm = 3 # parameter that limits the size of the weights
n_batch = len(x_train_raw) # batch size
input_dim = np.shape(x_train_raw)[1] # input dimension for autoencoder

### Scale training data ###
# After loading the training data, the data will likely need to be scaled. In
# this example, we scale each entry in the neighborhood graph from a range of 
# -1 to 1 (which is most appropriate for tanh activation functions).

min_list = [] # List of minimum values in training data
max_list = [] # List of maximum values in training data
x_train_scaled_list = []
for i in range(0, np.shape(x_train_raw)[1]):
    column_min = np.min(x_train_raw[:,i])
    column_max = np.max(x_train_raw[:,i])
    min_list.append(column_min)
    max_list.append(column_max)
    x_train_scaled_list.append(2*(x_train_raw[:,i]-column_min)/(column_max-column_min)-1)
x_train_scaled = np.transpose(np.asarray(x_train_scaled_list)) # Convert to numpy array  

# Convert min and max lists to arrays and save
min_array = np.asarray(min_list)
max_array = np.asarray(max_list)

np.save(path+'min_array.npy', min_array)
np.save(path+'max_array.npy', max_array)

### Create encoder ###
# First layer
encoder_initial = Dense(n_nodes, activation = activation_function, input_shape=(input_dim,), kernel_constraint=maxnorm(kc_max_norm))

# Hidden layers
encoder_list = []
for i in range(0,n_hidden_layers-1):
    encoder_list.append(Dense(n_nodes, activation = activation_function, kernel_constraint=maxnorm(kc_max_norm)))

# Final layer
encoder_final = Dense(bottleneck_size, activation = "linear")

### Create decoder ###
# First and hidden layers
decoder_list = []
for i in range(0,n_hidden_layers):
    decoder_list.append(Dense(n_nodes, activation = activation_function, kernel_constraint=maxnorm(kc_max_norm)))
    
# Final layer
decoder_final = Dense(input_dim, activation="linear")

### Create autoencoder ###
autoencoder = Sequential()
autoencoder.add(encoder_initial)
if use_dropout == True:
    autoencoder.add(Dropout(drop_prob))
for i in range(0, n_hidden_layers-1):
    autoencoder.add(encoder_list[i])
    if use_dropout == True:
        autoencoder.add(Dropout(drop_prob))
autoencoder.add(encoder_final)
for i in range(0, n_hidden_layers):
    autoencoder.add(decoder_list[i])
    if use_dropout == True:
        autoencoder.add(Dropout(drop_prob))
autoencoder.add(decoder_final)

### Train autoencoder model ###
# Note that the choice of optimizer and loss can change depending on user
# needs. Further note that "validation data" is included because it provides
# loss calculations that do not include dropout. Dropout is a tool to help 
# prevent over-fitting during training, yet is not used in the final
# autoencoder predictions
autoencoder.compile(optimizer='adam', loss='mean_squared_error')
autoencoder.fit(x_train_scaled, x_train_scaled,
                epochs=n_epoch,
                batch_size=n_batch,
                shuffle=True, validation_data=(x_train_scaled, x_train_scaled))

### Create encoder model ###
encoder = Sequential()
if use_dropout == False:
    for i in range(0, n_hidden_layers+1):
        encoder.add(autoencoder.layers[i])
else:
    encoder.add(autoencoder.layers[0])
    for i in range(1, n_hidden_layers+1):
        encoder.add(autoencoder.layers[2*i])

### Create and save low-dimensional representations of training data ###
lowd_train = encoder.predict(x_train_scaled)
np.save(path+"LowD_Train.npy", lowd_train)

### Save autoencoder ###
model_json_1 = autoencoder.to_json()
with open(path+"Autoencoder.json", "w") as json_file:
    json_file.write(model_json_1)

# serialize weights to HDF5
autoencoder.save_weights(path+"Autoencoder.h5")

### Save encoder ###
model_json_2 = encoder.to_json()
with open(path+"Encoder.json", "w") as json_file:
    json_file.write(model_json_2)

# serialize weights to HDF5
encoder.save_weights(path+"Encoder.h5")

### Plot loss ###
plt.figure(1)
plt.plot(autoencoder.history.history['val_loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig(path+"Autoencoder_Train.png")

### Save loss ###
np.save(path+'val_loss.png', autoencoder.history.history['val_loss'])

### Plot histograms of low-dimensional space ###
for i in range(0, np.shape(lowd_train)[1]):
    plt.figure(2+i)
    plt.hist(lowd_train[:, i], bins = 100)
    plt.xlabel('Order Parameter Value')
    plt.ylabel('Count')
    plt.title('Order Parameter #' + str(i))
    plt.savefig(path+"Order Parameter #" +str(i) +"_histogram.png")