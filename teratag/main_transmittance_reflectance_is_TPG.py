import numpy as np
from lib.Allread import allread
from lib.train_test_split import train_test_split
from lib.machine_learning.classification import svm,kNN,pCA
from lib.visualization import colorcode

y_all = []
flag = 0
#mainデータを読み込む。
for i in range(1,5):
    i = i*0.5
    for j in range(1,5):
        try:
            x = allread('reflectance').Frequency_trans_reflect_is_TPG('/Users/ryoya/kawaseken/20190201_fix/PE_{0}mm_{1}.txt'.format(i,j),
                '/Users/ryoya/kawaseken/20190201/ref.txt',1.4,1.5)
            if flag == 0:
                x_all = x
                flag += 1
            else:
                x_all = np.append(x_all, x, axis=0)
            #y_allの値がint出ないとsvm,pcaの可視化が上手くいかないので0.5mmの場合は*2などをして元に戻す。
            y_all.append(i*2)
        except FileNotFoundError as e:
            print(e)
#train_test_split(特徴量,目的関数,1つの厚さにおけるtrainデータの数)
train_x,train_y,test_x,test_y = train_test_split(x_all,y_all,1)

print(train_x)
print(train_y)
#print(test_x)
#print(test_y)
#print(x_all)
#print(y_all)
#referenceのカラーコード
#カラーコードのタグの数width=4,length=4の場合16個のタグに対応
width = 3
length = 4
colorcode(test_y,width,length)
#SVM
best_pred=svm(train_x,train_y,test_x,test_y)
colorcode(best_pred,width,length)
#k近傍法
best_pred=kNN(train_x,train_y,test_x,test_y)
colorcode(best_pred,width,length)
# PCA-SVM
transformed, targets = pCA(x_all, y_all)

train_x_pca,train_y_pca,test_x_pca,test_y_pca = train_test_split(transformed,targets,1)

best_pred = svm(train_x_pca, train_y_pca, test_x_pca, test_y_pca)
colorcode(best_pred, width, length)