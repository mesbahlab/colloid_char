### Import required packages ###
import numpy as np
import re

###############################################################################
# This script reads GDV files to extract all neighborhood graphs.
###############################################################################

### Enter path of GDV file of interest ###
df = "Trajectory_Data/Multi-Flavored_500_particles/cry_0.gdv"

data_list = []
for line in open(df, 'r'):
    item = line.rstrip()
    data_list.append(item)

data_string = ''.join(data_list)

#pattern = "\[ (.*?)\]"
pattern = "\[.+?\]"
signatures = re.findall(pattern, data_string)

for i in range(0, len(signatures)):
    signatures[i] = signatures[i].replace('[','')
    signatures[i] = signatures[i].replace(']','')

GDV_list = []
for i in range(0, len(signatures)):
    new_list = signatures[i].split(' ')
    new_list = [x for x in new_list if x]
    num_list = []
    for j in range(0, len(new_list)):
        num_list.append(int(new_list[j]))
    array = np.asarray(num_list)
    GDV_list.append(array)
    
GDV_array = np.asarray(GDV_list)  

### Save array of neighborhood graphs###
np.save("Results/Neigbhorhood_Graphs/Multi-Flavored_500_particles/cry_0.npy", GDV_array)
















