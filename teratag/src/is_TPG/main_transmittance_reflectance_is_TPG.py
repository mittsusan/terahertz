import numpy as np
import sys
sys.path.append('../../')
from lib import allread
from lib import train_test_split,decide_test_number
#from sklearn.model_selection import train_test_split
from lib import svm,kNN,pCA,svm_gridsearchcv,randomforest,gaussiannb
from lib import colorcode
#######測定物の度に変更して下さい
date_dir = '/Users/ryoya/kawaseken'
shielding_material = '/cardboard2_denim2'
sensitivity = 'nosensitivity'
from_frequency = 1.0
to_frequency = 1.8
frequency_list = [] #周波数を指定しない場合は空にして下さい。
inten_or_trans_or_reflect = 0 #0の時強度、1の時透過率、2の時反射率
#mainデータを読み込む。
last_type = 6 #使用する種類
last_num = 10 #最後の種類の使用するファイル数
#カラーコードのタグの数width=4,length=4の場合16個のタグに対応
width = 3
length = 7
test_number = 3
pca_third_argument = 1 #PCAの第3引数で0の場合厚み、それ以外は糖類になるように設定。
#######

file_name_list = [] #filenameの保持
thickness = 'mm'
y_all = []
flag = 0

#データの読み込み
for i in range(1,last_type+1):
    #ここで厚みの選択及び糖
    #i = i*0.5
    for j in range(1,last_num+1):
        try:
            x = allread(inten_or_trans_or_reflect,str(i)+thickness,i,j,last_type,last_num,from_frequency,to_frequency,frequency_list).Frequency_trans_reflect_is_TPG(date_dir + shielding_material + '/' + sensitivity + '/' + str(i) + '/' + str(j) + '.txt',
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

#gaussianNB
print('\nGaussianNB')
best_pred=gaussiannb(train_x,train_y,test_x,test_y)
colorcode(best_pred,width,length)
#RF
print('\nRF')
best_pred=randomforest(train_x,train_y,test_x,test_y,from_frequency,to_frequency,frequency_list)
colorcode(best_pred,width,length)
#SVM
print('\nSVM')
best_pred=svm(train_x,train_y,test_x,test_y)
colorcode(best_pred,width,length)
#SVM_grid_searchCV
'''
print('\nSVM_gridsearch_CV')
best_pred=svm_gridsearchcv(train_x,train_y,test_x,test_y)
colorcode(best_pred,width,length)
'''
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