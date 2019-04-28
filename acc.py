import numpy as np


def load_data(filename):
    dataset = []
    with open(filename, 'rt') as f:
        for line in f:
            dataset.append(float(line.strip()))
    return dataset

def acc(dataset_name,cluster1_name, cluster2_name):
    dataset = load_data(dataset_name)
    data_c1 = set(dataset[:100])
    data_c2 = set(dataset[100:200])

    cluster1= set(load_data(cluster1_name))
    cluster2= set(load_data(cluster2_name))

    diff1 =  len(cluster1.intersection(data_c1))
    diff2 =  len(cluster2.intersection(data_c2))
    acc1 = diff1 / len(cluster1)
    acc2 = diff2 / len(cluster2)
    print(' c1 accuracy = %f' %acc1)
    print(' c2 accuracy = %f' %acc2)


dataset_name = 'dataset.txt'    #read dataset from textfile.
print('accuracy of k-means: ')
acc(dataset_name,'k-means-cluster-1.txt', 'k-means-cluster-2.txt')
print('\n')
print('accuracy of EM:')
acc(dataset_name,'EM-cluster-1.txt', 'EM-cluster-2.txt')
