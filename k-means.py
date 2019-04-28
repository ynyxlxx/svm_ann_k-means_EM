import random
import math
import argparse
import time

parser = argparse.ArgumentParser(description='k-means')
parser.add_argument('dataset_name')
args = parser.parse_args()

def save_result(filename, cluster):
    file = open(filename, 'w')
    for point in cluster:
        file.write(str(point) + '\n')
    file.close()
    return

    #read dataset from textfile.
dataset = []
with open(args.dataset_name, 'rt') as f:
    for line in f:
        dataset.append(float(line.strip()))

#initialization
max_number_in_data = max(dataset)
min_number_in_data = min(dataset)

start_centroid_1 = random.uniform(min_number_in_data, max_number_in_data)
start_centroid_2 = random.uniform(min_number_in_data, max_number_in_data)

#distance computation
t = time.time()
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

print('time cost: %fs' %(time.time()-t))
save_result('k-means-cluster-1.txt', cluster_1)
save_result('k-means-cluster-2.txt', cluster_2)

print('save result to text file k-means-cluster1.txt and k-means-cluster2.txt.')