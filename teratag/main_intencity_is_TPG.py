import numpy as np
import glob
import os
import matplotlib
import matplotlib.pyplot as plt
from lib.Allread import allread
from lib.train_test_split import train_test_split
from lib.machine_learning.classification import svm,kNN,pCA
from lib.visualization import colorcode
#from sklearn.model_selection import train_test_split
from sklearn import preprocessing

plt.close()
l = 1
y_all = []
flag = 0
med = [ 'グルコース', 'ラクトース']
#試薬の数だけ繰り返すように組んでいこう
#mainデータを読み込む。
#ここでディレクトリの移動をするので開きたいファイルのパスを入力してください
#uuuuu
os.chdir('/Users/toshinari/Downloads/暫定')
for w in med:
    for j in glob.glob("{0}*.txt".format(w)):
        try:
            x = allread('Intencity').Frequency_Intencity_is_TPG("/Users/toshinari/Downloads/暫定/{0}".format(j), 1.05, 1.8)
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
mm = preprocessing.MinMaxScaler()
x_all_2 = mm.fit_transform(x_all)
print(x_all_2)
print(x_all)
#train_test_split(特徴量,目的関数,1つの厚さにおけるtrainデータの数)
train_x,train_y,test_x,test_y = train_test_split(x_all_2,y_all,1)
print(y_all)
#print(train_x)
print(train_y)

#print(test_x)
#print(test_y)
#print(x_all)
#print(y_all)
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
#transformed, targets = pCA(x_all_2, y_all)

#train_x_pca,train_y_pca,test_x_pca,test_y_pca = train_test_split(transformed,targets,1)

#best_pred = svm(train_x_pca, train_y_pca, test_x_pca, test_y_pca)
#colorcode(best_pred, width, length)
