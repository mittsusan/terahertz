import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
import sys
sys.path.append('../../')
from sklearn.model_selection import train_test_split
from lib.machine_learning.classification import svm,kNN
from lib.Change_Trans import Max_Min
from lib.Change_Trans import pCA,iCA
from sklearn.pipeline import make_pipeline
from skrebate import ReliefF
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from lib.Change_Trans import Trans_file

#データを大量に読み込んでラベルづけを行うプログラム
plt.close()
y_all = []
flag = 0
l = 0
i = 0
k = 0
m = 0
#教師データのある大元フォルダの選択
#path_1 = '/Users/toshinari/Downloads/SVM_train'

#num = 1#教師データのフォルダの数(適宜変更)

#med = ['lac_2', 'lac_3', 'lac_4', 'lac_5', 'lac_6']#対象の指定
med = ['lac','mal','glu']

Teacher = os.chdir('/Users/toshinari/Downloads/A_RESALT/ONE_shield')
directory = os.listdir(Teacher)
#print(directory)
directory = sorted(directory)
print(directory)
len_dir = len(directory)
#print(len_dir)
first = 0.8
last = 2.0
l=0
#print(directory[2])
for i in range(0,len_dir-1):
    choice_dir = directory[i+1]
    os.chdir('/Users/toshinari/Downloads/A_RESALT/ONE_shield/{}'.format(choice_dir))

    dir_list = sorted(os.listdir('/Users/toshinari/Downloads/A_RESALT/ONE_shield/{}'.format(choice_dir)))#ディレクトリをソートして取得

    #print(dir_list)#それぞれの日付フォルダ内のファイルおよびディレクトリからなるリスト
    #for l in range(0,len(dir_list)-1):

    try:
        for j in med:
            #print(j)
            for k in sorted(glob.glob("{0}*.txt".format(j))):
                df = pd.read_csv(k, engine='python', header=None, index_col=0, sep='\t')
                #以下２行は別ファイルで定義した関数
                trans = Trans_file(k,"ref_s.txt")
                #print(type(trans))
                trans.to_csv("/Users/toshinari/Downloads/A_RESALT/Trans/{0}/Trans_{1}_{2}.csv".format(j,k.rstrip(".txt"),choice_dir), sep = ",")
                #os.remove('/Users/toshinari/Downloads/SVM_file/INPUT/{}/{}'.format(choice_dir, k))
    except FileNotFoundError as e:
        print(e)
    l = l+1
    print(l)



plt.close()

#それぞれの試薬のフォルダを読み込んで教師データとしてラベリング
first = 1.1
last = 1.7
x_list = []
y_list = []
l=0
j = 0
from sklearn.model_selection import train_test_split

sample_list = ['mal','lac','glu']

#以下教師データのラベルづけ
for w in sample_list:
    path_1 = os.chdir('/Users/toshinari/Downloads/A_RESALT/Trans/{}'.format(w))

    file_list = sorted(glob.glob('/Users/toshinari/Downloads/A_RESALT/Trans/{}/*'.format(w)))
    #print(file_list)
    file_number = len(file_list)
    print(file_number)
    #a = np.empty((121,file_number))
    for k in range(0,file_number):
        df = pd.read_csv(file_list[k], engine='python', header=None, index_col=0, sep=',')

        df = df[first:last]#特定の周波数範囲の抜き取り

        #print(df)
        df = Max_Min(df)#正規化
        #print(df.iloc[:,0])
        #ここまで欲しいところを抜き出している過程
        plt.plot(df)
        #plt.show()
        plt.close()
        df_np = df.values
        #print(df_np)
        if k ==0:
            x_all = df_np

            #j = j + 1
        else:
            x_all = np.append(x_all,df_np,axis = 1)
        y_list.append(l)
    #x_all = np.array([x_list])
    if l==0:
        X_all = x_all

    else:
        X_all = np.append(X_all,x_all, axis = 1)
    l = l+1

#print(y_list)
y_all = np.array(y_list)
#print(X_all.T)
#print(y_all.shape)


X_train, X_test, y_train, y_test = train_test_split(X_all.T,y_all, test_size = 0.2)
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









#以下測定データのラベル付け





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