import numpy as np
import pandas as pd
import glob
import os
import os.path
import matplotlib.pyplot as plt
import sys
sys.path.append('../../')
from lib.Med_def_file import Trans_file

#データを大量に読み込んで遮蔽物毎にサンプルをフォルダ分けするプログラム
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

med = ['lac','mal','glu']
File = ['Intencity', 'Trans']


Teacher = os.chdir('/Users/toshinari/Downloads/SVM_file/INPUT_2')#遮蔽物の種類毎にサンプルを分類したいとき用
directory = os.listdir(Teacher)
#print(directory)
directory = sorted(directory)
print(directory)
len_dir = len(directory)
#print(len_dir)
first = 0.8
last = 2.0
l=0
#print(directory[2])
#以下MacOSでの特殊な処理。MacOSではフォルダ内に複数のフォルダが存在すると謎のファイルDS_Storeが自動で生成される。Windowsで実行する場合はfor文のrange内の-1を消す。そしてchoice_dirの+1を消す
for i in range(0,len_dir-1):
    date_dir = directory[i+1]
    print(date_dir)
    os.chdir("/Users/toshinari/Downloads/SVM_file/INPUT_2/{}".format(date_dir))
    #print(os.getcwd())
    dir_list = sorted(os.listdir('/Users/toshinari/Downloads/SVM_file/INPUT_2/{}'.format(date_dir)))#ディレクトリをソートして取得

    #print(dir_list)#それぞれの日付フォルダ内のファイルおよびディレクトリからなるリスト
    #for l in range(0,len(dir_list)-1):

    shield_list = [n for n in dir_list if not n .count('.txt')]
    #print(n)
    print(shield_list)
    #len_shield_list = len(shield_list)

    os.chdir('/Users/toshinari/Downloads/SVM_file/INPUT_2/{}/{}'.format(date_dir,shield_list[1]))
    print(os.getcwd())

    if os.path.exists('/Users/toshinari/Downloads/SVM_file/ALL_RESULT/{}/'.format(shield_list[1])) == False:
        os.mkdir('/Users/toshinari/Downloads/SVM_file/ALL_RESULT/{}/'.format(shield_list[1]))
        os.chdir('/Users/toshinari/Downloads/SVM_file/ALL_RESULT/{}/'.format(shield_list[1]))
        os.mkdir(File[0])
        os.mkdir(File[1])
    else:
        os.chdir('/Users/toshinari/Downloads/SVM_file/ALL_RESULT/{}/'.format(shield_list[1]))
        print('Exist')

    print(os.getcwd())
    print(len(File))
    for m in range(0,len(File)):
        os.chdir('/Users/toshinari/Downloads/SVM_file/ALL_RESULT/{}/'.format(shield_list[1]))
        os.chdir('{}'.format(File[m]))
        try:
            for j in med:
                os.chdir('/Users/toshinari/Downloads/SVM_file/ALL_RESULT/{}/{}'.format(shield_list[1], File[m]))
                #os.chdir()
                if os.path.exists('{}'.format(j)) == False:
                    os.mkdir('{}'.format(j))
                #print(j)
                os.chdir('/Users/toshinari/Downloads/SVM_file/INPUT_2/{}/{}'.format(date_dir,shield_list[1]))
                for k in sorted(glob.glob("{0}*.txt".format(j))):
                #for k in range(0,len(os.listdir('/Users/toshinari/Downloads/SVM_file/INPUT_2/{}/{}')):
                    df = pd.read_csv(k, engine='python', header=None, index_col=0, sep=',')
                    print('OK')
                    #os.chdir('/Users/toshinari/Downloads/SVM_file/ALL_RESULT')
                    #os.mkdir('/Users/toshinari/Downloads/SVM_file/ALL_RESULT/{}/Intencity/{}'.format(shield_list[i+1],j))
                    #os.mkdir('/Users/toshinari/Downloads/SVM_file/ALL_RESULT/{}/Trans/{}'.format(shield_list[i+1],j))
                    df.to_csv('/Users/toshinari/Downloads/SVM_file/ALL_RESULT/{0}/Intencity/{1}/Intencity_{2}_{3}.csv'.format(shield_list[1],j,k.rstrip(".txt"),date_dir), sep = ",")
                    #print(k)
                    #以下２行は別ファイルで定義した関数
                    trans = Trans_file(k,'ref_s.txt',first,last)
                    #print(type(trans))
                    trans.to_csv('/Users/toshinari/Downloads/SVM_file/ALL_RESULT/{0}/Trans/{1}/Trans_{2}_{3}.csv'.format(shield_list[1],j,k.rstrip(".txt"),date_dir), sep = ",")
                    os.remove('/Users/toshinari/Downloads/SVM_file/INPUT_2/{}/{}/{}'.format(date_dir,shield_list[1], k))
        except FileNotFoundError as e:
            print(e)
        l = l+1
        print(l)
'''

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
'''
