from scipy import io
import collections
import numpy as np
import time
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn import svm

def generate_data_index_dict(label_array):
    dict = collections.defaultdict(list)
    index = 0
    for i in label_array:
        dict[i[0]].append(index)
        index += 1
    return dict

def extract_data(data_set, data_size, label_number, label_dict, label_data_set):
    train_data = []
    label_data = []
    j = 0
    for i in label_dict[label_number]:
        if j < data_size:
            train_data.append(data_set[i])
            label_data.append(label_data_set[i])
            j += 1

    return train_data, label_data

def get_test_set(label_dict, data_set, label_data_set, label_number, data_size):
    index = 0
    test_data = []
    test_label = []
    for i in reversed(label_dict[label_number]):
        if index < data_size:
            test_data.append(data_set[i])
            test_label.append(label_data_set[i])
            index += 1

    return test_data, test_label


mnist = io.loadmat('mnist.mat')

data, label = mnist["data"].T/255, mnist["label"].T

dict = generate_data_index_dict(label)

############################################# 100 200 500 1000 2000 5000

train_data_100_1, label_data_100_1 = extract_data(data, data_size=1000,label_number=1,label_dict=dict, label_data_set=label)
train_data_200_1, label_data_200_1 = extract_data(data, data_size=2000,label_number=1,label_dict=dict, label_data_set=label)
train_data_500_1, label_data_500_1 = extract_data(data, data_size=200,label_number=1,label_dict=dict, label_data_set=label)
train_data_1000_1, label_data_1000_1  = extract_data(data, data_size=500,label_number=1,label_dict=dict, label_data_set=label)


train_data_100_9, label_data_100_9 = extract_data(data, data_size=4000,label_number=9,label_dict=dict, label_data_set=label)
train_data_200_9, label_data_200_9 = extract_data(data, data_size=3000,label_number=9,label_dict=dict, label_data_set=label)
train_data_500_9, label_data_500_9 = extract_data(data, data_size=4800,label_number=9,label_dict=dict, label_data_set=label)
train_data_1000_9, label_data_1000_9 = extract_data(data, data_size=4500,label_number=9,label_dict=dict, label_data_set=label)

test_data_1, test_label_1 = get_test_set(dict, data, label, 1, 1000)
test_data_9, test_label_9 = get_test_set(dict, data, label, 9, 1000)

train_data = np.vstack((train_data_100_1, train_data_100_9))
train_label = np.vstack((label_data_100_1, label_data_100_9))
test_data = np.vstack((test_data_1, test_data_9))
test_label = np.vstack((test_label_1,test_label_9))

print(np.shape(train_data))
print(np.shape(train_label))
print(np.shape(test_data))
print(np.shape(test_label))

t = time.time()

mlp = MLPClassifier(solver='sgd', activation='relu', alpha=1e-4, hidden_layer_sizes=(100, ), random_state=1,
                        max_iter=10, verbose=10, learning_rate_init=.1)

mlp.fit(train_data, train_label)
pre = mlp.predict(test_data)

print(classification_report(test_label, pre))
acc = accuracy_score(test_label, pre)
print('label 1 : 1000, 4000 ' + '准确率：%f,花费时间：%.2fs' % (acc, time.time() - t))

# t = time.time()
#
# svc = svm.SVC(kernel = 'rbf',C = 1.0)
# svc.fit(train_data, train_label)
# pre = svc.predict(test_data)
#
# print(classification_report(test_label, pre))
# acc= accuracy_score(test_label, pre)
# print ('label 1 : 500,4500 ' + u'准确率：%f,花费时间：%.2fs' %(acc,time.time()-t))