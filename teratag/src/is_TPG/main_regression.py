import numpy as np
import sys
sys.path.append('../../')
from lib import allread
from lib import train_test_split,decide_test_number,decide_test_number_multi_regressor
#from sklearn.model_selection import train_test_split
from lib import svm,kNN,pCA,svm_gridsearchcv,randomforest,gaussiannb
from lib import colorcode
from lib import ridge_multi,svr_linear_multi,svr_rbf_multi,randomforest_regression,dnn,keras_dnn,keras_dnn_predict
date_dir = '/Users/ryoya/kawaseken/20191201'
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
file_name_list = [] #filenameの保持
thickness = 'mm'
flag = 0

nb_epoch = 7000
nb_batch = 32
learning_rate = 1e-1
dense1 = 60
dense2 = 30
dense3 = 14
dense4 = class_number =last_type
regularizers_l2_1 = 0
regularizers_l2_2 = 0
regularizers_l2_3 = 0
model_structure = '{0}relul2{1}_{2}relul2{3}_{4}relul2{5}_{6}softmax'.format(dense1,regularizers_l2_1,dense2,regularizers_l2_2,dense3,regularizers_l2_3,dense4)

#データの読み込み
for i in range(1,last_type+1):

    for concentration in range(1, concentration_pattern+1):
        y = np.zeros(last_type)
        if i == last_type:
            y[i - 1] = 100 - 20 * (concentration - 1)
            y[0] = 0 + 20 * (concentration - 1)
        else:
            y[i - 1] = 100 - 20 * (concentration - 1)
            y[i] = 0 + 20 * (concentration - 1)
        y = np.array([y])

        for j in range(1,last_num+1):
            try:
                x = allread(inten_or_trans_or_reflect,str(i)+thickness,i*concentration,j,last_type,last_num,from_frequency,to_frequency,frequency_list).Frequency_trans_reflect_is_TPG(date_dir
                    + shielding_material + '/' + sensitivity + '/' + str(i) + '/' + str(concentration) + '/'+ str(j) + '.txt',
                        date_dir + shielding_material + '/ref.txt')

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

# print(type(train_x))
# print(type(train_y))
# print(test_x)
# print(test_y)
# print(x_all)
# print(y_all)
#referenceのカラーコード

#colorcode(test_y,width,length)


# #Ridge回帰
# print('\nRidge回帰')
# best_pred=ridge_multi(train_x,train_y,test_x,test_y)
# #colorcode(best_pred,width,length)
#
# #SVR線形回帰
# print('\nSVR線形回帰')
# best_pred=svr_linear_multi(train_x,train_y,test_x,test_y)
# #colorcode(best_pred,width,length)
#
# #SVRガウシアン回帰
# print('\nSVRガウシアン回帰')
# best_pred=svr_rbf_multi(train_x,train_y,test_x,test_y)
# #colorcode(best_pred,width,length)


print('\nランダムフォレスト回帰')
best_pred=randomforest_regression(train_x,train_y,test_x,test_y,from_frequency,to_frequency,frequency_list)
#colorcode(best_pred,width,length)

# print('\nニューラルネット回帰')
# best_pred=dnn(train_x,train_y,test_x,test_y)
# #colorcode(best_pred,width,length)

print('\nニューラルネット回帰(Kerasで最適化)')
best_pred=keras_dnn(train_x,train_y,test_x,test_y,from_frequency,to_frequency,frequency_list,last_type,shielding_material,
                    nb_epoch, nb_batch, learning_rate,dense1, dense2, dense3, dense4, regularizers_l2_1, regularizers_l2_2, regularizers_l2_3)
# keras_dnn_predict(train_x,train_y,test_x,test_y,from_frequency,to_frequency,frequency_list,last_type,shielding_material,
#                   nb_epoch, nb_batch, learning_rate,dense1, dense2, dense3, dense4, regularizers_l2_1, regularizers_l2_2, regularizers_l2_3,model_structure)
#colorcode(best_pred,width,length)