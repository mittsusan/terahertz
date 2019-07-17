import pandas as pd
import glob
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
import numpy as np
import scipy.stats as stats
import os


#全てのファイルにラベルをつける（教師データの作成）
def Label_Sample_File(sample_list,first,last):
    y_list = []
    l = 0
    for w in sample_list:
        #path_1 = os.chdir('/Users/toshinari/Downloads/SVM_file/ALL_RESULT/ALL_file/Trans/{}'.format(w))

        file_list = sorted(glob.glob('{}/*'.format(w)))
        print(file_list)
        file_number = len(file_list)
        print(file_number)
        # a = np.empty((121,file_number))
        for k in range(0, file_number):
            df = pd.read_csv(file_list[k], engine='python', header=None, index_col=0, sep=',')
            df = df[first:last]  # 特定の周波数範囲の抜き取り
            # print(df)
            df = Max_Min(df)  # 正規化
            # print(df.iloc[:,0])
            # ここまで欲しいところを抜き出している過程
            df_np = df.values
            # print(df_np)
            if k == 0:
                x_all = df_np

                # j = j + 1
            else:
                x_all = np.append(x_all, df_np, axis=1)
            y_list.append(l)
        # x_all = np.array([x_list])
        if l == 0:
            X_all = x_all

        else:
            X_all = np.append(X_all, x_all, axis=1)
        l = l + 1

    # print(y_list)
    y_all = np.array(y_list)

    return y_all, X_all, file_list


def Trans_file(file, ref):
    df = pd.read_csv(file, engine='python', header=None, index_col=0, sep=',')
    df_ref = pd.read_csv(ref, engine='python', header=None, index_col=0, sep=',')
    # ここで強度を透過率に変化
    df.iloc[:, 0] = df.iloc[:, 0] / df_ref.iloc[:, 0]
    trans = df
    # self.Frequency_trans_reflect_is_TPG_FFT(0) #振幅スペクトルが欲しい場合はnumberを0、位相スペクトルが欲しい時はnumberを1
    return trans


def Graph_Trans_file(file,first,last):
    df = pd.read_csv(file, engine='python', header=None, index_col=0, sep=',')
    df = df[first:last]
    # ここで強度を透過率に変化
    df = df.iloc[:, 0]
    trans = Max_Min(df)
    '''
    plt.style.use('ggplot')
    font = {'family': 'meiryo'}
    matplotlib.rc('font', **font)
    '''

    #plt.plot()
    trans.plot()
    #plt.xlabel('周波数[THz]')
    plt.title(file.lstrip('/Users/toshinari/Downloads/SVM_file/ALL_RESULT/ALL_file/Trans/'))

    plt.show()
    plt.close()

    # self.Frequency_trans_reflect_is_TPG_FFT(0) #振幅スペクトルが欲しい場合はnumberを0、位相スペクトルが欲しい時はnumberを1
    return


def Max_Min(df):
    list_index = list(df.index)
    #print(df)
    x = df.values
    # print(x)
    min = x.min(axis=None, keepdims=True)
    max = x.max(axis=None, keepdims=True)
    result = (x - min) / (max - min)
    #print(result)
    x_1 = pd.DataFrame(result, index=list_index)
    #df.iloc[:,0] = x_1.iloc[:,0]
    #x_1  = df
    #print(x_1)

    return x_1


def pCA(x_all, y_all):
    # 主成分分析する
    features = x_all
    targets = y_all
    pca = PCA(n_components=2)
    pca.fit(features)
    # 分析結果を元にデータセットを主成分に変換する
    transformed = pca.fit_transform(features)
    #print(transformed.shape)
    #print(len(targets))
    # print(transformed)
    # 主成分をプロットする
    for label in np.unique(y_all):
        plt.scatter(transformed[y_all == label, 0],
                    transformed[y_all == label, 1], )
    plt.legend(loc='upper right',
               bbox_to_anchor=(1,1),
               borderaxespad=0.5,fontsize = 10)
    plt.title('principal component')
    plt.xlabel('Pc1')
    plt.ylabel('Pc2')

    # 主成分の寄与率を出力する
    #print('各次元の寄与率: {0}'.format(decomposer.explained_variance_ratio_))
    #print('累積寄与率: {0}'.format(sum(decomposer.explained_variance_ratio_)))
    print(pca.explained_variance_ratio_)

    # グラフを表示する
    plt.show()



    return transformed, targets


def iCA(x_all, y_all):
    # 独立成分の数＝24
    decomposer = FastICA(n_components=2)
    # データの平均を計算
    M = np.mean(x_all, axis=1)[:, np.newaxis]
    # 各データから平均を引く
    data2 = x_all - M
    # 平均0としたデータに対して、独立成分分析を実施
    decomposer.fit(data2)

    # 独立成分ベクトルを取得(D次元 x 独立成分数)
    S = decomposer.transform(data2)
    #プロットする
    for label in np.unique(y_all):
        plt.scatter(S[y_all == label, 0],
                    S[y_all == label, 1], )
    plt.legend(loc='upper right',
               bbox_to_anchor=(1,1),
               borderaxespad=0.5,fontsize = 10)
    plt.title(' component')
    plt.xlabel('Ic1')
    plt.ylabel('Ic2')

    # 主成分の寄与率を出力する
    #print('各次元の寄与率: {0}'.format(decomposer.explained_variance_ratio_))
    #print('累積寄与率: {0}'.format(sum(decomposer.explained_variance_ratio_)))

    # グラフを表示する
    plt.show()


    return S, y_all

'''
    if number == 0:
        for label in np.unique(targets):
            plt.scatter(transformed[targets == label, 0],
                        transformed[targets == label, 1], label='{}mm'.format(label*0.5))
    else:
        for label in np.unique(targets):
            if label == 1:
                plt.scatter(transformed[targets == label, 0],
                            transformed[targets == label, 1], label='Glucose')
            elif label == 2:
                plt.scatter(transformed[targets == label, 0],
                            transformed[targets == label, 1], label='Lactose')
            elif label == 3:
                plt.scatter(transformed[targets == label, 0],
                            transformed[targets == label, 1], label='Maltose')
            elif label == 4:
                plt.scatter(transformed[targets == label, 0],
                            transformed[targets == label, 1], label='Glu_Lac')
            elif label == 5:
                plt.scatter(transformed[targets == label, 0],
                            transformed[targets == label, 1], label='Lac_Mal')
            elif label == 6:
                plt.scatter(transformed[targets == label, 0],
                            transformed[targets == label, 1], label='Mal_Glu')
            elif label == 7:
                plt.scatter(transformed[targets == label, 0],
                            transformed[targets == label, 1], label='Glu_Lac_Mal')

    plt.legend(loc='upper right',
               bbox_to_anchor=(1,1),
               borderaxespad=0.5,fontsize = 10)
    plt.title('principal component')
    plt.xlabel('pc1(a.u.)')
    plt.ylabel('pc2(a.u.)')

    # 主成分の寄与率を出力する
    print('各次元の寄与率: {0}'.format(pca.explained_variance_ratio_))
    print('累積寄与率: {0}'.format(sum(pca.explained_variance_ratio_)))

    # グラフを表示する
    plt.show()
'''
'''
def PCA_found(file):#PCA出力し、プロットする関数
    pca = PCA(n_components=2)
    pca.fit(file)
    transformed = pca.fit_transform(file)
    plt.scatter(transformed[])

'''



'''

    plt.scatter(transformed[y_all == label, 0],
                transformed[y_all == label, 1], )
plt.legend(loc='upper right',
           bbox_to_anchor=(1, 1),
           borderaxespad=0.5, fontsize=10)
plt.title('principal component')
plt.xlabel('Ic1')
plt.ylabel('Ic2')

# 主成分の寄与率を出力する
# print('各次元の寄与率: {0}'.format(decomposer.explained_variance_ratio_))
# print('累積寄与率: {0}'.format(sum(decomposer.explained_variance_ratio_)))

# グラフを表示する
plt.show()
'''