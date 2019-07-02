import numpy as np
import glob
import os
import matplotlib
import matplotlib.pyplot as plt
import sys
sys.path.append('../../')
from lib.Allread import allread
from lib.train_test_split import train_test_split
from lib.machine_learning.classification import svm,kNN,pCA,iCA
from lib.visualization import colorcode
#from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd


plt.close()
l = 1
y_all = []
flag = 0
med = ['mal', 'lac']
#試薬の数だけ繰り返すように組んでいこう
#mainデータを読み込む。
#ここでディレクトリの移動をするので開きたいファイルのパスを入力してください
#uuuuu
path_1 = '/Users/toshinari/Downloads/OneDrive_1_2019-6-19/20190619/加藤さん遮蔽物'

os.chdir(path_1)
for w in med:
    for j in glob.glob("{0}_*.txt".format(w)):
        try:
            x, ref = allread('Trans').Frequency_trans_reflect_is_TPG("{0}".format(j),"ref_s1_1.txt", 1.05, 1.8)
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

df_med = pd.read_table("/Users/toshinari/Downloads/OneDrive_1_2019-6-19/20190619/加藤さん遮蔽物/L_1.5mm_1.txt", engine='python',index_col=0)
df_med = df_med[1.0:1.8]
ref = ref[1.0:1.8]
ref_2 = df_med.iloc[:,0] / ref.iloc[:,0]
#ref_2 = df_med.iloc[:,0] / ref.iloc[:.0]
#np_ref = ref.values
#np_med = df_med.values
##print(np_med.shape)
#print(np_ref.shape)
#ref_2 = np_med[1.0:1.8,:] / np_ref[1.0:1.8,:]
#print(ref_2)
ref_3 = ref_2.values

mm = preprocessing.MinMaxScaler()
x_all_2 = mm.fit_transform(x_all)

#得られたテキストに書き込み



#print(x_all_2)
#print(x_all_2)
#train_test_split(特徴量,目的関数,1つの厚さにおけるtrainデータの数)
train_x,train_y,test_x,test_y = train_test_split(x_all_2,y_all,1)
#print(y_all)
print(train_x)
#print(train_y)
#print(type(train_x))
print(test_x)
#print(test_y)
#referenceのカラーコード
#カラーコードのタグの数width=4,length=4の場合16個のタグに対応
width = 3
length = 4
#colorcode(test_y,width,length)
#SVM
best_pred=svm(train_x,train_y,test_x,test_y)
#print(best_pred)
#print('a')
#colorcode(best_pred,width,length)
#k近傍法
#best_pred=kNN(train_x,train_y,test_x,test_y)
#colorcode(best_pred,width,length)

# PCA-SVM
transformed, targets = pCA(x_all_2, y_all)

train_x_pca,train_y_pca,test_x_pca,test_y_pca = train_test_split(transformed,targets,3)

best_pred = svm(train_x_pca, train_y_pca, test_x_pca, test_y_pca)
#colorcode(best_pred, width, length)


#ICA-SVM
S, y_all = iCA(x_all_2, y_all)
train_x_ica,train_y_ica,test_x_ica,test_y_ica = train_test_split(S,y_all,3)

best_pred = svm(train_x_ica, train_y_ica, test_x_ica, test_y_ica)