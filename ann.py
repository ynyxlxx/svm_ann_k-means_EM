import numpy as np
from scipy import io
from sklearn.svm import SVC
from sklearn import svm
import time
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_mldata
import numpy as np
import pickle
import gzip
from sklearn.metrics import classification_report

np.set_printoptions(threshold=np.inf)

data = io.loadmat('mnist.mat')
print(list(data))

x, y = data["data"].T/255, data["label"].T



train_data = x[0:60000]
train_label = y[0:60000]


shuffle_index = np.random.permutation(60000)  # permutation函数是用来打乱数据顺序的函数
train_data, train_label = train_data[shuffle_index], train_label[shuffle_index]

t_num=[]
t_num=[2000,5000,10000,20000,30000,40000,50000]
test_num = 1000

for i in t_num:
    train_num = i

    tr_data = train_data[:train_num]
    tr_label = train_label[:train_num]
    te_data = train_data[train_num:train_num + test_num]
    te_label = train_label[train_num:train_num + test_num]
    te_label = te_label.reshape((test_num, 1))
    t = time.time()

    mlp = MLPClassifier(solver='sgd', activation='relu', alpha=1e-4, hidden_layer_sizes=(100, ), random_state=1,
                        max_iter=10, verbose=10, learning_rate_init=.1)

    mlp.fit(tr_data, tr_label)
    pre = mlp.predict(te_data)

    print(classification_report(te_label, pre))
    acc = accuracy_score(te_label, pre)
    print('训练集大小: '+str(train_num)+u',准确率：%f,花费时间：%.2fs' % (acc, time.time() - t))