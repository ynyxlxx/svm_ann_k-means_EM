import numpy as np
from scipy import io
from sklearn.svm import SVC
from sklearn import svm
import sys
from sklearn.decomposition import PCA
import time
from sklearn.metrics import accuracy_score

np.set_printoptions(threshold=np.inf)

data = io.loadmat('mnist.mat')
print(list(data))


x, y = data["data"].T / 255, data["label"].T

print('data matrix: ' + str(np.shape(x)))
print('data matriy: ' + str(np.shape(y)))

train_num = 50000
test_num = 60000

train_data = x[0:60000]
train_label = y[0:60000]
print('train_data: ' + str(np.shape(train_data)))
print('train_lable: ' + str(np.shape(train_label)))



shuffle_index = np.random.permutation(60000) # permutation函数是用来打乱数据顺序的函数
train_data, train_label = train_data[shuffle_index], train_label[shuffle_index]

tr_data = train_data[:train_num]
tr_label = train_label[:train_num]
te_data = train_data[train_num:test_num]
te_label = train_label[train_num:test_num]
te_label = te_label.reshape((test_num - train_num, 1))
t = time.time()



svc = svm.SVC(kernel = 'rbf',C = 1.0)
svc.fit(tr_data,tr_label)
pre = svc.predict(te_data)


acc= accuracy_score(te_label,pre)
print (u'准确率：%f,花费时间：%.2fs' %(acc,time.time()-t))
