import random
import math
import numpy as np
import time
import argparse

parser = argparse.ArgumentParser(description='EM')
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

pi=3.1415926

u_a = 3.5
u_b = 5.5

u_a_1 = 6
u_b_1 = 10

o_a = 1.5
o_b = 1.5

p_a = 0.5
p_b = 0.5

p_x_a = np.zeros((1, len(dataset)))
p_x_b  = np.zeros((1, len(dataset)))
a = np.zeros((1, len(dataset)))
b = np.zeros((1, len(dataset)))
differ_a = np.zeros((1, len(dataset)))
differ_b = np.zeros((1, len(dataset)))

t = time.time()
counter = 1
while True:
    print('iteration: %i' %counter)
    counter += 1
    # computing the posterior probability (E step)
    for i in range(len(dataset)):
        x_i = dataset[i]
        differ_a[0][i] = (x_i - u_a)**2
        differ_b[0][i] = (x_i - u_b)**2
        p_x_a[0][i] = math.exp(-((x_i - u_a) ** 2) / (2 * o_a)) / math.sqrt(2 * pi * o_a)
        p_x_b[0][i] = math.exp(-((x_i - u_b) ** 2) / (2 * o_b)) / math.sqrt(2 * pi * o_b)

        a[0][i] = p_x_a[0][i] * p_a / (p_x_a[0][i] * p_a + p_x_b[0][i] * p_b)
        b[0][i] = p_x_b[0][i] * p_b / (p_x_a[0][i] * p_a + p_x_b[0][i] * p_b)
    # M step
    o_a = np.average(differ_a[0], weights=a[0])
    u_a_1 = np.average(dataset, weights=a[0])
    o_b = np.average(differ_b[0], weights=b[0])
    u_b_1 = np.average(dataset, weights=b[0])
    p_a = np.mean(a[0])
    p_b = np.mean(b[0])



    if abs(u_a_1-u_a)>0.05 and abs(u_b_1-u_b)>0.05:
        u_a = u_a_1
        u_b = u_b_1
        print('μ1 update : %f' %u_a)
        print('μ2 update : %f' % u_b)
        print('σ1 update : %f' % math.sqrt(o_a))
        print('σ2 update : %f' % math.sqrt(o_b))

    else:
        print('final μ1 :  %f' %u_a_1)
        print('final μ2 :  %f' %u_b_1)
        print('final σ1 :  %f' % math.sqrt(o_a))
        print('final σ2 :  %f' % math.sqrt(o_b))
        u1=u_a_1
        u2=u_b_1
        break


cluster_1 = []
cluster_2 = []
for i in range(len(dataset)):
    x_i = dataset[i]
    p_x_a[0][i] = math.exp(-((x_i - u_a) ** 2) / (2 * o_a)) / math.sqrt(2 * pi * o_a)
    p_x_b[0][i] = math.exp(-((x_i - u_b) ** 2) / (2 * o_b)) / math.sqrt(2 * pi * o_b)

    if p_x_a[0][i] > p_x_b[0][i]:
        cluster_1.append(x_i)
    else:
        cluster_2.append(x_i)

print('time cost: %fs' %(time.time()-t))
save_result('EM-cluster-1.txt', cluster_1)
save_result('EM-cluster-2.txt', cluster_2)

print('save result to text file EM-cluster-1.txt and EM-cluster-2.txt.')

