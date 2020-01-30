import numpy as np
import sys
sys.path.append('../../')
from lib import allread
from lib import train_test_split_madebymitsuhashi,decide_test_number,decide_test_number_onehot
#from sklearn.model_selection import train_test_split
from lib import svm,kNN,pCA,svm_gridsearchcv,randomforest,gaussiannb,dnn_classification
from lib import colorcode,concentration_colorcode
from lib import ridge,svr_linear,svr_rbf
from keras.utils import np_utils


# カラーコードのタグの数width=4,length=4の場合16個のタグに対応
width = 3
length = 10
test_number = 1
pca_third_argument = 1  # PCAの第3引数で0の場合厚み、それ以外は糖類になるように設定。
class_number = 30
date_dir = '/Users/ryoya/kawaseken/20200127_for_analysis'
shielding_material = '/syntheticleather_leather'
sensitivity = '/50'
base_dir = date_dir + shielding_material + sensitivity
from_frequency = 1.0
to_frequency = 1.5
frequency_list = []
class Read:
    def __init__(self):
        #######測定物の度に変更して下さい
        self.date_dir = date_dir
        self.shielding_material = shielding_material
        self.sensitivity = sensitivity
        self.from_frequency = from_frequency
        self.to_frequency = to_frequency
        self.frequency_list = frequency_list  # 周波数を指定しない場合は空にして下さい。
        self.inten_or_trans_or_reflect = 1  # 0の時強度、1の時透過率、2の時反射率
        self.last_type = class_number  # 使用する種類
        self.last_num = 4  # 最後の種類の使用するファイル数
        self.add = 1  # フォルダを新しく追加した場合そのフォルダの数　1つの場合は1
        self.file_name_list = []  # filenameの保持
        self.type_name_list = []  # 試薬及び厚みの保持
        self.concentration_color = [0,0,0]
        self.y_all = []
        self.y_all_dnn = []
        self.flag = 0
        self.flag2 = 0
    def read(self):
        global x_all
        for i in range(1,self.last_type+1):
            #ここで厚みの選択及び糖
            #i = i*0.5
            self.type_name_list.append(i)
            for j in range(1,self.last_num+1):
                try:
                    x = allread(self.inten_or_trans_or_reflect,i,j,self.last_type,self.last_num,self.from_frequency,
                                self.to_frequency,self.frequency_list).Frequency_trans_reflect_is_TPG(self.date_dir + self.shielding_material + self.sensitivity + '/' + str(i) + '/' + str(j) + '.txt',
                            self.date_dir + self.shielding_material + '/ref.txt')

                    self.file_name_list.append(j)

                    if self.flag == 0:
                        x_all = x
                        self.flag += 1

                    else:
                        x_all = np.append(x_all, x, axis=0)


                    #y_allの値がint出ないとsvm,pcaの可視化が上手くいかないので0.5mmの場合は*2などをして元に戻す。
                    self.y_all.append(i)
                    self.y_all_dnn.append(i-1)
                except FileNotFoundError as e:
                    print(e)

        return x_all, self.y_all, self.file_name_list, self.type_name_list, self.y_all_dnn

    def read_conbineoldwithnew(self):

        # データの読み込み
        global concentration_color_type
        concentration_color_type = []
        for i in range(1, self.last_type + 1):
            # ここで厚みの選択及び糖
            # i = i*0.5
            self.type_name_list.append(i)
            concentration_interval = 0.1
            type1 = 10
            type2 = 20
            type3 = 30
            if i <= type1:
                self.concentration_color[0] = int(255 * concentration_interval * (type1 - (i - 1)))
                self.concentration_color[1] = int(255 * concentration_interval * (i - 1))
            elif i == type1 + 1:
                self.concentration_color[0] = 0
                self.concentration_color[1] = 255
                self.concentration_color[2] = 0
            elif i <= type2:
                self.concentration_color[1] = int(255 * concentration_interval * (type2 - (i - 1)))
                self.concentration_color[2] = int(255 * concentration_interval * (i - 1 - type1))
            elif i == type2 + 1:
                self.concentration_color[0] = 0
                self.concentration_color[1] = 0
                self.concentration_color[2] = 255
            elif i <= type3:
                self.concentration_color[2] = int(255 * concentration_interval * (type3 - (i - 1)))
                self.concentration_color[0] = int(255 * concentration_interval * (i - 1 - type2))



            if self.flag2 == 0:
                concentration_color_type = np.array([self.concentration_color])
                self.flag2 += 1

            else:
                concentration_color_type = np.append(concentration_color_type, np.array([self.concentration_color]), axis=0)


            for k in range(1, self.add + 1):
                for j in range(1, self.last_num + 1):
                    try:
                        x = allread(self.inten_or_trans_or_reflect, i, j, self.last_type, self.last_num, self.from_frequency,
                                    self.to_frequency, self.frequency_list).Frequency_trans_reflect_is_TPG(
                            self.date_dir + self.shielding_material + self.sensitivity + '/' + str(i) + '/' + str(k) + '/' + str(j) + '.txt',
                            self.date_dir + self.shielding_material + self.sensitivity + '/' + str(i) + '/' + str(k) + '/' + 'ref.txt')

                        self.file_name_list.append(j)

                        if self.flag == 0:
                            x_all = x
                            self.flag += 1

                        else:
                            x_all = np.append(x_all, x, axis=0)

                        # y_allの値がint出ないとsvm,pcaの可視化が上手くいかないので0.5mmの場合は*2などをして元に戻す。
                        self.y_all.append(i)
                        self.y_all_dnn.append(i-1)
                    except FileNotFoundError as e:
                        print(e)
        #print(concentration_color_type)

        return x_all, self.y_all, self.file_name_list, self.type_name_list, concentration_color_type, self.y_all_dnn



read = Read()

x_all, y_all, file_name_list, type_name_list, concentration_color_type, y_all_dnn = read.read_conbineoldwithnew()
#train_test_split(特徴量,目的関数,1つの厚さにおけるtrainデータの数)
#train_x,train_y,test_x,test_y = train_test_split_madebymitsuhashi(x_all,y_all,1)
train_x,train_y,test_x,test_y = decide_test_number(x_all,y_all,test_number)
#train_x, test_x, train_y, test_y = train_test_split(x_all, y_all, test_size=3)

#print(type(train_x))
#print(type(train_y))
#print(test_x)
#print(test_y)
#print(x_all)
#print(y_all)

#referenceのカラーコード
concentration_colorcode(test_y, width, length, concentration_color_type)

# #gaussianNB
# print('\nGaussianNB')
# best_pred=gaussiannb(train_x,train_y,test_x,test_y)
# colorcode(best_pred,width,length)
#
# #RF
# print('\nRF')
# best_pred=randomforest(train_x,train_y,test_x,test_y,from_frequency,to_frequency,frequency_list)
# colorcode(best_pred,width,length)



#SVM
print('\nSVM')
best_pred=svm(train_x, train_y, test_x, test_y)
concentration_colorcode(best_pred, width, length, concentration_color_type)

# #SVM_grid_searchCV
#
# print('\nSVM_gridsearch_CV')
# best_pred=svm_gridsearchcv(train_x,train_y,test_x,test_y)
# colorcode(best_pred,width,length)

#k近傍法
print('\nK近傍法')
best_pred=kNN(train_x, train_y, test_x, test_y)
concentration_colorcode(best_pred, width, length, concentration_color_type)
# PCA-SVM
print('\nPCA-SVM')

transformed, targets = pCA(x_all, y_all, pca_third_argument, file_name_list, type_name_list, concentration_color_type)

train_x_pca, train_y_pca, test_x_pca, test_y_pca = decide_test_number(transformed, targets, test_number)

best_pred = svm(train_x_pca, train_y_pca, test_x_pca, test_y_pca)
concentration_colorcode(best_pred, width, length, concentration_color_type)

## ここからDNN
Y_ = np_utils.to_categorical(np.array(y_all_dnn), class_number)
#train_x,train_y,test_x,test_y = train_test_split(x_all,Y_,1)#trainを選択する場合
train_x,train_y,test_x,test_y = decide_test_number_onehot(x_all,Y_,test_number)#testを選択する場合

best_pred, probability = dnn_classification(train_x, train_y, test_x, test_y, class_number, base_dir, from_frequency, to_frequency, frequency_list)
print('\nDNN')
print('best_pred:{}'.format(best_pred))
print('probability:{}'.format(probability))
print('average_probability:{}'.format(sum(probability)/len(probability)))
concentration_colorcode(best_pred, width, length, concentration_color_type)
