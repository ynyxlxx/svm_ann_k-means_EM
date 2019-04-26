import numpy as np

dataset_name = 'dataset.txt'    #read dataset from textfile.
dataset = []
with open(dataset_name, 'rt') as f:
    for line in f:
        dataset.append(float(line.strip()))

print(dataset)

#initialization



#distance computation


#assignment of a data point to a cluster

#update of the centroid

