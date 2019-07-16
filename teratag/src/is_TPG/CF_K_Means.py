import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import sys
sys.path.append('../../')
from lib.Allread import allread
from sklearn.cluster import KMeans
from lib.train_test_split import train_test_split
from lib.machine_learning.classification import svm,kNN,pCA

#from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.cluster import KMeans

#データを大量に読み込んでラベルづけを行うプログラム

y_all = []
flag = 0
l = 0
#使いたいフォルダの選択
path_1 = '/Users/toshinari/Downloads/SVM_train/SVM_train_3'
plt.close()

#med = ['lac_2', 'lac_3', 'lac_4', 'lac_5', 'lac_6']#対象の指定
med = ['lac']
os.chdir(path_1)

for w in med:
    for j in sorted(glob.glob("{0}*.txt".format(w))):

        try:
            x, ref = allread('Trans','a','b','c', 'd', 'e').Frequency_trans_reflect_is_TPG("{0}".format(j),"ref_s.txt", 1.05, 1.8)
            print(j)
            if flag == 0:
                x_all = x
                flag += 1
            else:
                x_all = np.append(x_all, x, axis=0)

            y_all.append(l)
        except FileNotFoundError as e:
            print(e)
l = l + 1



pred = KMeans(n_clusters=2).fit_predict(x_all)
y_all = np.array(y_all)
print(pred)
print(y_all)





