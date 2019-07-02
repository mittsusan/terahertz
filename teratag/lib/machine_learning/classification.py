from sklearn.svm import SVC,SVR
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn import neighbors
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.decomposition import FastICA

def svm(train_x, train_y, test_x, test_y):
    param_list = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    best_score = 0
    best_parameters = {}
    kernel = 'rbf'
    for gamma in param_list:  # グリッドサーチをしてハイパーパラメータ探索
        for C in param_list:
            estimator = SVC(gamma=gamma, kernel=kernel, C=C)
            classifier = OneVsRestClassifier(estimator)
            classifier.fit(train_x, train_y)
            pred_y = classifier.predict(test_x)
            classifier2 = SVC(C=C, kernel=kernel, gamma=gamma)
            classifier2.fit(train_x, train_y)
            pred_y2 = classifier2.predict(test_x)
            onerest_score = accuracy_score(test_y, pred_y)
            oneone_score = accuracy_score(test_y, pred_y2)
            if onerest_score > oneone_score:
                score = onerest_score
                hikaku = 'One-versus-the-rest'
                better_pred = pred_y
            else:
                score = oneone_score
                hikaku = 'One-versus-one'
                better_pred = pred_y2
            # 最も良いスコアのパラメータとスコアを更新
            if score > best_score:
                best_hikaku = hikaku
                best_score = score
                best_parameters = {'gamma': gamma, 'C': C}
                best_pred = better_pred

    print('Best score: {}'.format(best_score))
    print('Best parameters: {}'.format(best_parameters))
    print('比較方法:{}'.format(best_hikaku))
    print('Best pred:{}'.format(best_pred))

    return best_pred


def kNN(train_x, train_y, test_x, test_y):
    k_list = [1]  # k の数（今回は訓練データが1つずつなので、1のみ）
    weights_list = ['uniform', 'distance']  # 今回は訓練データが一つなので、このパラメータは関係なくなる。
    for weights in weights_list:
        for k in k_list:
            clf = neighbors.KNeighborsClassifier(k, weights=weights)
            clf.fit(train_x, train_y)
            # 正答率を求める
            pred_y = clf.predict(test_x)
            ac_score = metrics.accuracy_score(pred_y, test_y)
            # print(type(k))
            # print(type(iris_y_test))
            print('k={0},weight={1}'.format(k, weights))
            print('正答率 =', ac_score)

    return pred_y


def pCA(x_all, y_all,number):
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
    plt.title('principal component')
    plt.xlabel('Ic1')
    plt.ylabel('Ic2')

    # 主成分の寄与率を出力する
    #print('各次元の寄与率: {0}'.format(decomposer.explained_variance_ratio_))
    #print('累積寄与率: {0}'.format(sum(decomposer.explained_variance_ratio_)))

    # グラフを表示する
    plt.show()


    return S, y_all