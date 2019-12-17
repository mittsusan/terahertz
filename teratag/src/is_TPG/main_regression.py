import numpy as np
import sys
sys.path.append('../../')
from lib import allread
from lib import train_test_split,decide_test_number,decide_test_number_multi_regressor
#from sklearn.model_selection import train_test_split
from lib import svm,kNN,pCA,svm_gridsearchcv,randomforest,gaussiannb
from lib import colorcode
from lib import ridge_multi,svr_linear_multi,svr_rbf_multi
#######測定物の度に変更して下さい
date_dir = '/Users/kawaselab/PycharmProjects/20191201'
shielding_material = '/syntheticleather_leather_regression'
sensitivity = '50'
dir = date_dir + shielding_material + '/' + sensitivity
from_frequency = 0.9
to_frequency = 1.6
frequency_list = [] #周波数を指定しない場合は空にして下さい。
inten_or_trans_or_reflect = 0 #0の時強度、1の時透過率、2の時反射率
#mainデータを読み込む。
last_type = 3 #使用する試薬の種類
concentration_pattern = 5 #使用する濃度のパターン
last_num = 5 #最後の種類の使用するファイル数
#カラーコードのタグの数width=4,length=4の場合16個のタグに対応
width = 3
length = 5
test_number = 1
pca_third_argument = 1 #PCAの第3引数で0の場合厚み、それ以外は糖類になるように設定。
#######
y = np.zeros(3)
file_name_list = [] #filenameの保持
thickness = 'mm'
flag = 0

#データの読み込み
for i in range(1,last_type+1):
    for concentration in range(1, concentration_pattern+1):
        for j in range(1,last_num+1):
            try:
                x = allread(inten_or_trans_or_reflect,str(i)+thickness,i*concentration,j,last_type,last_num,from_frequency,to_frequency,frequency_list).Frequency_trans_reflect_is_TPG(date_dir
                    + shielding_material + '/' + sensitivity + '/' + str(i) + '/' + str(concentration) + '/'+ str(j) + '.txt',
                        date_dir + shielding_material + '/ref.txt')
                if i == last_type:
                    y[i-1] = 100 - 20*(concentration - 1)
                    y[0] =  0 + 20*(concentration - 1)
                else:
                    y[i-1] == 100 - 20*(concentration - 1)
                    y[i] == 0 + 20*(concentration - 1)
                file_name_list.append(j)

                if flag == 0:
                    x_all = x
                    y_all = y
                    flag += 1

                else:
                    x_all = np.append(x_all, x, axis=0)
                    y_all = np.append(y_all, y, axis=0)

                #y_allの値がint出ないとsvm,pcaの可視化が上手くいかないので0.5mmの場合は*2などをして元に戻す。
            except FileNotFoundError as e:
                print(e)

#train_test_split(特徴量,目的関数,1つの厚さにおけるtrainデータの数)
#train_x,train_y,test_x,test_y = train_test_split(x_all,y_all,1)
train_x,train_y,test_x,test_y = decide_test_number_multi_regressor(x_all,y_all,test_number)
#train_x, test_x, train_y, test_y = train_test_split(x_all, y_all, test_size=3)

print(type(train_x))
print(type(train_y))
print(test_x)
print(test_y)
#print(x_all)
#print(y_all)
#referenceのカラーコード

#colorcode(test_y,width,length)


#Ridge回帰
print('\nRidge回帰')
best_pred=ridge_multi(train_x,train_y,test_x,test_y)
#colorcode(best_pred,width,length)

#SVR線形回帰
print('\nSVR線形回帰')
best_pred=svr_linear_multi(train_x,train_y,test_x,test_y)
#colorcode(best_pred,width,length)

#SVRガウシアン回帰
print('\nSVRガウシアン回帰')
best_pred=svr_rbf_multi(train_x,train_y,test_x,test_y)
#colorcode(best_pred,width,length)
