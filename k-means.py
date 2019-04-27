import random
import math
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
max_number_in_data = max(dataset)
min_number_in_data = min(dataset)

start_centroid_1 = random.uniform(min_number_in_data, max_number_in_data)
start_centroid_2 = random.uniform(min_number_in_data, max_number_in_data)

#distance computation
counter = 1
while True:
    print('iteration: %i' %counter)
    counter += 1
    cluster_1 = []
    cluster_2 = []
    for num in dataset:
        distance_cluster_1 = math.sqrt(pow(start_centroid_1 - num, 2))
        distance_cluster_2 = math.sqrt(pow(start_centroid_2 - num, 2))

        if distance_cluster_1 < distance_cluster_2:  #assignment of a data point to a cluster
            cluster_1.append(num)
        else:
            cluster_2.append(num)

    new_centroid_1 = sum(cluster_1) / len(cluster_1) #update of the centroid
    new_centroid_2 = sum(cluster_2) / len(cluster_2)

    if start_centroid_1 == new_centroid_1 and start_centroid_2 == new_centroid_2:
        break
    else:
        start_centroid_1 = new_centroid_1
        start_centroid_2 = new_centroid_2

save_result('k-means-cluster-1.txt', cluster_1)
save_result('k-means-cluster-2.txt', cluster_2)

print('save result to text file.')