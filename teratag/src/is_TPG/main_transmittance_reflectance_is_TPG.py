import numpy as np
import sys
sys.path.append('../../')
from lib.Allread import allread
from lib.train_test_split import train_test_split
from lib.machine_learning.classification import svm,kNN,pCA
from lib.visualization import colorcode

y_all = []
flag = 0
#mainデータを読み込む。
num = 5 #numは使用する最後のファイル名の数＋１(rangeのため)
for i in range(2,num):
    #ここで厚みの選択
    i = i*0.5
    for j in range(1,6):
        if j <= 4:
            try:
                x = allread('振幅[a.u.]','{}mm'.format(i),num-1).Frequency_trans_reflect_is_TPG(r'C:\Users\kawaselab\PycharmProjects\mitsuhashi\shahei\{}zink\{}.txt'.format(i,j),
                    r'C:\Users\kawaselab\PycharmProjects\mitsuhashi\shahei\ref.txt',1.4,1.6)

                if flag == 0:
                    x_all = x
                    flag += 1

                else:
                    x_all = np.append(x_all, x, axis=0)


                #y_allの値がint出ないとsvm,pcaの可視化が上手くいかないので0.5mmの場合は*2などをして元に戻す。
                y_all.append(i*2)
            except FileNotFoundError as e:
                print(e)
        #ここに訓練データを追加していく形で。
        '''
        else:
            try:
                x = allread('reflectance','{}mm'.format(i)).Frequency_trans_reflect_is_TPG('/Users/ryoya/kawaseken/20190207_fix/PE_{0}mm_{1}.txt'.format(i,j),
                    '/Users/ryoya/kawaseken/20190207_fix/ref.txt',1.4,1.6)

                if flag == 0:
                    x_all = x
                    flag += 1

                else:
                    x_all = np.append(x_all, x, axis=0)


                #y_allの値がint出ないとsvm,pcaの可視化が上手くいかないので0.5mmの場合は*2などをして元に戻す。
                y_all.append(i*2)
            except FileNotFoundError as e:
                print(e)
        '''

#train_test_split(特徴量,目的関数,1つの厚さにおけるtrainデータの数)
train_x,train_y,test_x,test_y = train_test_split(x_all,y_all,1)

#print(train_x)
#print(train_y)
#print(test_x)
#print(test_y)
#print(x_all)
#print(y_all)
#referenceのカラーコード
#カラーコードのタグの数width=4,length=4の場合16個のタグに対応
width = 3
length = 3
colorcode(test_y,width,length)
#SVM
print('\nSVM')
best_pred=svm(train_x,train_y,test_x,test_y)
colorcode(best_pred,width,length)
#k近傍法
print('\nK近傍法')
best_pred=kNN(train_x,train_y,test_x,test_y)
colorcode(best_pred,width,length)
# PCA-SVM
print('\nPCA-SVM')
transformed, targets = pCA(x_all, y_all)

train_x_pca,train_y_pca,test_x_pca,test_y_pca = train_test_split(transformed,targets,1)

best_pred = svm(train_x_pca, train_y_pca, test_x_pca, test_y_pca)
colorcode(best_pred, width, length)