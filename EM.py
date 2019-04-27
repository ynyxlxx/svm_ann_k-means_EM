import random
import math
import numpy as np
import argparse

def save_result(filename, cluster):
    file = open(filename, 'w')
    for point in cluster:
        file.write(str(point) + '\n')
    file.close()
    return

dataset_name = 'dataset.txt'    #read dataset from textfile.
dataset = []
with open(dataset_name, 'rt') as f:
    for line in f:
        dataset.append(float(line.strip()))

#initialization
mean_start_1 = 1.5
mean_start_2 = 2.5

sigma_start_1 = 3
sigma_start_2 = 8

Pc1 = 0.4
Pc2 = 0.6
#computing the posterior probability (E step)
prob_point_c1 = np.zeros((1, len(dataset)))
prob_point_c2 = np.zeros((1, len(dataset)))
prob_c1_point = np.zeros((1, len(dataset)))
prob_c2_point = np.zeros((1, len(dataset)))

for i in range(len(dataset)): #p(xi|c) p(c|xi)
    point = dataset[i]



print(prob_point_c1)