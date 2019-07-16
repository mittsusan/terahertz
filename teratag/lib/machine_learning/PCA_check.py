import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
import sys
sys.path.append('../../')
from sklearn.model_selection import train_test_split
from lib.machine_learning.classification import svm, kNN
from lib.Med_def_file  import Max_Min
from lib.Med_def_file import pCA,iCA
import seaborn as sns
from sklearn.neighbors import LocalOutlierFactor
from lib.Med_def_file import Graph_Trans_file
from lib.Med_def_file import Label_Sample_File

plt.close()

# それぞれの試薬のフォルダを読み込んで教師データとしてラベリング

path = '/Users/toshinari/Downloads/OneDrive_1_2019-6-27/20190621/aodan+danb'
first = 1.05
last = 1.8
x_list = []
y_list = []
l = 0
j = 0

from sklearn.model_selection import train_test_split

sample_list = ['lac']
path_1 = os.chdir('/Users/toshinari/Downloads/SVM_file/ALL_RESULT/ALL_file/Trans')

y_all, X_all,file_list = Label_Sample_File(sample_list,first,last)


#print(X_all.T)
print(y_all.shape)

print('\nPCA')
A, B = pCA(X_all.T, y_all)
print('\nICA')
C,D = iCA(X_all.T,y_all)
clf = LocalOutlierFactor(n_neighbors=7, contamination=0.01)
pred = clf.fit_predict(X_all.T)

# 正常データのプロット
#plt.scatter(A[:, 0][np.where(pred > 0)], A[:, 1][np.where(pred > 0)])
# 異常データのプロット
#plt.scatter(A[:, 0][np.where(pred < 0)], A[:, 1][np.where(pred < 0)])

#以下エラーファイルの表示
k = [idx for idx,val in enumerate(pred) if val<=0]
print(k)
for i in range(0,len(k)):
    error_file_num = (k[i])
    print(error_file_num)
    print(file_list[error_file_num])
    f_path = file_list[error_file_num]
    Graph_Trans_file(f_path, first, last)

#plt.show()
# 以下測定データのラベル付け


'''
for w in sample_list:
    os.chdir('/Users/toshinari/Downloads/SVM_file/ALL_RESULT/ALL_file/Trans/{}'.format(w))
    file_list = sorted(glob.glob('/Users/toshinari/Downloads/OneDrive_1_2019-6-27/20190624/*.txt'))

    for k in range(0,file_number):
        df = pd.read_csv(file_list[k], engine='python', header=None, index_col=0, sep='\t')
        #print(type(df))
        df = df[first:last]#特定の周波数範囲の抜き取り
        df = Max_Min(df)#正規化
        x_list.append(df.iloc[:,0])
        y_list.append(l)
    x_all = np.array([x_list])
    print(x_all)
    print(y_list)
    l = l+1
'''

# = svm(x_all,y_all,x_test,y_test)
'''
df = pd.read_csv('')

# self.graph_Frequency_trans_reflect_is_TPG()
# print(self.df)
for j in self.df.iloc[:, 0]:
    x_list.append(j)

# [1.12, 1.23, 1.3, 1.36, 1.45, 1.55, 1.6]
x_all = np.array([x_list])

return x_all
'''