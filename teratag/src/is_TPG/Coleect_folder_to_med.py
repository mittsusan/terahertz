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
path_1 = '/Users/toshinari/Downloads/SVM_train'

#num = 1#教師データのフォルダの数(適宜変更)

#med = ['lac_2', 'lac_3', 'lac_4', 'lac_5', 'lac_6']#対象の指定
med = ['lac','mal','glu']

Teacher = os.chdir('/Users/toshinari/Downloads/SVM_file/INPUT')
directory = os.listdir(Teacher)
print(directory)
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
    os.chdir('/Users/toshinari/Downloads/SVM_file/INPUT/{}'.format(choice_dir))

    dir_list = sorted(os.listdir('/Users/toshinari/Downloads/SVM_file/INPUT/{}'.format(choice_dir)))#ディレクトリをソートして取得

    #print(dir_list)#それぞれの日付フォルダ内のファイルおよびディレクトリからなるリスト
    #for l in range(0,len(dir_list)-1):

    try:
        for j in med:
            #print(j)
            for k in sorted(glob.glob("{0}*.txt".format(j))):
                df = pd.read_csv(k, engine='python', header=None, index_col=0, sep='\t')
                df.to_csv("/Users/toshinari/Downloads/SVM_file/ALL_RESULT/ALL_file/Intencity/{0}/Intencity_{1}_{2}.csv".format(j,k.rstrip(".txt"),choice_dir), sep = ",")
                #print(k)
                #以下２行は別ファイルで定義した関数
                trans = Trans_file(k,"ref_s.txt")
                #print(type(trans))
                trans.to_csv("/Users/toshinari/Downloads/SVM_file/ALL_RESULT/ALL_file/Trans/{0}/Trans_{1}_{2}.csv".format(j,k.rstrip(".txt"),choice_dir), sep = ",")
                #os.remove('/Users/toshinari/Downloads/SVM_file/INPUT/{}/{}'.format(choice_dir, k))
    except FileNotFoundError as e:
        print(e)
    l = l+1
    print(l)


"""
        if os.path.isdir('/Users/toshinari/Downloads/SVM_file/INPUT/{}/{}'.format(choice_dir,dir_list[l+1])) == True:#読み込んだものがディレクトリだったならば
            os.chdir('/Users/toshinari/Downloads/SVM_file/INPUT/{}/{}'.format(choice_dir,dir_list[l+1]))
            #print(os.path.dirname('/Users/toshinari/Downloads/SVM_file/INPUT/{}/{}'.format(choice_dir,dir_list[l+1])))
            print(dir_list[l+1])
            os.mkdir('/Users/toshinari/Downloads/SVM_file/ALL_RESULT/{}'.format(dir_list[l+1]))

            try:
                for j in med:
                    # print(j)
                    os.mkdir('/Users/toshinari/Downloads/SVM_file/ALL_RESULT/{}/{}'.format(dir_list[l + 1], j))

                    for k in sorted(glob.glob("{0}*.txt".format(j))):
                        df = pd.read_csv(k, engine='python', header=None, index_col=0, sep='\t')
                        df.to_csv(
                            "/Users/toshinari/Downloads/SVM_file/ALL_RESULT/ALL_file/Intencity/{0}/Intendity_{1}.csv".format(
                                j, k.rstrip(".txt")), sep=",")
                        # print(k)
                        # 以下２行は別ファイルで定義した関数
                        trans = Trans_file(k, "ref_s.txt")
                        # print(type(trans))
                        trans.to_csv(
                            "/Users/toshinari/Downloads/SVM_file/ALL_RESULT/ALL_file/Trans/{0}/Trans_{1}.csv".format(
                                j,k.rstrip(".txt")),sep=",")
                        # os.remove('/Users/toshinari/Downloads/SVM_file/INPUT/{}/{}'.format(choice_dir, k))
            except FileNotFoundError as e:
                print(e)

        else:
            print('b')
"""

"""

dir_list = os.listdir('/Users/toshinari/Downloads/SVM_file/INPUT/{}'.format(choice_dir))
                if dir_list[]:
                df = pd.read_csv(k, engine = 'python', header = None,index_col=0,sep = '\t')
                df_ref = pd.read_csv("ref_s.txt", engine = 'python', header = None, index_col=0, sep = '\t')
                #freq = df.iloc[:,0]
                df.iloc[:,0] = df.iloc[:,0]/df_ref.iloc[:,0]
                print(df)
                df = df[first:last]
                #print(df)
                x = df.values
                #print(x)
                min = x.min(axis=None, keepdims=True)
                max = x.max(axis=None, keepdims=True)
                result = (x - min) / (max - min)
                print(result)
                x_1 = pd.DataFrame(result)
                print(x_1)
    """