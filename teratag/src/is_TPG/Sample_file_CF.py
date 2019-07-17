import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
import sys
sys.path.append('../../')
from sklearn.model_selection import train_test_split
from lib.machine_learning.classification import svm,kNN
from lib.Med_def_file import Max_Min
from lib.Med_def_file import pCA,iCA
from sklearn.pipeline import make_pipeline
from skrebate import ReliefF
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from lib.Med_def_file import Label_Sample_File#ファイルのラベルづけを行う関数



plt.close()

#それぞれの試薬のフォルダを読み込んで教師データとしてラベリング

path ='/Users/toshinari/Downloads/OneDrive_1_2019-6-27/20190621/aodan+danb'
first = 1.1
last = 1.8
x_list = []
y_list = []
l=0
j = 0

from sklearn.model_selection import train_test_split

sample_list = ['glu','mal','lac']
path = os.chdir('/Users/toshinari/Downloads/SVM_file/ALL_RESULT/ALL_file/Trans')
#以下教師データのラベルづけ

#ファイル全体のラベルづけを行う関数
y_all, X_all,file_list = Label_Sample_File(sample_list,first,last)
#print(X_all.T)
#print(y_all.shape)


X_train, X_test, y_train, y_test = train_test_split(X_all.T,y_all, test_size = 0.3)
#print(X_test)
print(y_test)
print('\nSVM')
best_pred=svm(X_train,y_train,X_test,y_test)
print('\nK近傍法')
best_pred=kNN(X_train,y_train,X_test,y_test)

print('\nPCA')
A,B = pCA(X_all.T,y_all)

print('\nICA')
C,D = iCA(X_all.T,y_all)

clf = RandomForestClassifier(n_estimators=40, random_state=42)
clf.fit(X_train, y_train)

#予測データ作成
y_predict = clf.predict(X_test)

#正解率
print('\n正答率')
print(accuracy_score(y_test, y_predict))

print(y_test)
print(y_predict)
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