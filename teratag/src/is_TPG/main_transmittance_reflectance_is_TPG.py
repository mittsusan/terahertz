import numpy as np
import sys
sys.path.append('../../')
from lib.Allread import allread
from lib.train_test_split import train_test_split,decide_test_number
#from sklearn.model_selection import train_test_split
from lib.machine_learning.classification import svm,kNN,pCA
from lib.visualization import colorcode

file_name_list = [] #filenameの保持
date_dir = '/Users/ryoya/kawaseken'
shielding_material = '/EMS_Si_reflect'
sensitivity = 'nosensitivity'
from_frequency = 1.4
to_frequency = 1.6
frequency_list = [] #周波数を指定しない場合は空にして下さい。
thickness = 'mm'
y_axis = 'reflectance'
y_all = []
flag = 0
#mainデータを読み込む。
last_type = 3 #使用する種類
last_num = 4 #最後の種類の使用するファイル数
#カラーコードのタグの数width=4,length=4の場合16個のタグに対応
width = 3
length = 3
test_number = 3
pca_third_argument = 0 #PCAの第3引数で0の場合厚み、それ以外は糖類になるように設定。

#データの読み込み
for i in range(1,last_type+1):
    #ここで厚みの選択及び糖
    #i = i*0.5
    for j in range(1,last_num+1):
        try:
            x = allread(y_axis,str(i)+thickness,i,j,last_type,last_num,from_frequency,to_frequency,frequency_list).Frequency_trans_reflect_is_TPG(date_dir + shielding_material + '/' + sensitivity + '/' + str(i) + '/' + str(j) + '.txt',
                    date_dir + shielding_material + '/ref.txt')

            file_name_list.append(j)

            if flag == 0:
                x_all = x
                flag += 1

            else:
                x_all = np.append(x_all, x, axis=0)


            #y_allの値がint出ないとsvm,pcaの可視化が上手くいかないので0.5mmの場合は*2などをして元に戻す。
            y_all.append(i)
        except FileNotFoundError as e:
            print(e)

#train_test_split(特徴量,目的関数,1つの厚さにおけるtrainデータの数)
#train_x,train_y,test_x,test_y = train_test_split(x_all,y_all,1)
train_x,train_y,test_x,test_y = decide_test_number(x_all,y_all,test_number)
#train_x, test_x, train_y, test_y = train_test_split(x_all, y_all, test_size=3)

print(type(train_x))
print(type(train_y))
#print(test_x)
#print(test_y)
#print(x_all)
#print(y_all)
#referenceのカラーコード

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

transformed, targets = pCA(x_all, y_all, pca_third_argument,file_name_list)

train_x_pca,train_y_pca,test_x_pca,test_y_pca = decide_test_number(transformed,targets,test_number)

best_pred = svm(train_x_pca, train_y_pca, test_x_pca, test_y_pca)
colorcode(best_pred, width, length)