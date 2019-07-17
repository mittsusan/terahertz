from sklearn.svm import SVC,SVR
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn import neighbors
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.decomposition import FastICA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import scipy.stats as stats
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

def randomforest(train_x, train_y, test_x, test_y, from_frequency, to_frequency, frequency_list):
    # use a full grid over all parameters
    param_grid = {"n_estimators": np.arange(50,300,10)}

    forest_grid = GridSearchCV(estimator=RandomForestClassifier(random_state=0,bootstrap = True),
                               param_grid=param_grid,
                               cv=3)
    forest_grid.fit(train_x, train_y)  # fit
    forest_grid_best = forest_grid.best_estimator_  # best estimator
    best_pred = forest_grid_best.predict(test_x)
    best_score = accuracy_score(best_pred, test_y)
    print("Best Model Parameter: ", forest_grid.best_params_)
    print('Best score: {}'.format(best_score))
    # 特徴量の重要度
    feature_importances = forest_grid_best.feature_importances_
    plt.figure(figsize=(10, 5))
    y = feature_importances
    x = np.arange(from_frequency,to_frequency+0.01,0.01)
    plt.bar(x, y, width = 0.005, align="center")
    plt.xlabel('frequency[THz]')
    plt.ylabel('feature importance')
    plt.show()

    return best_pred


def svm(train_x, train_y, test_x, test_y):
    param_list = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000,10000]
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

def svm_gridsearchcv(train_x, train_y, test_x, test_y):
    ### 探索するパラメータ空間
    def param():
        ret = {
            'C': np.linspace(0.0001, 1000, 10),
            'kernel': ['rbf', 'linear', 'poly'],
            'degree': np.arange(1, 6, 1),
            'gamma': np.linspace(0.0001, 1000, 10)
        }
        return ret
    # GridSearchCVのインスタンスを作成&学習&スコア記録
    gscv_one = GridSearchCV(SVC(), param(), cv=3,return_train_score=False, verbose=0)
    gscv_one.fit(train_x, train_y)
    # 最高性能のモデルを取得
    best_one_vs_one = gscv_one.best_estimator_
    best_pred_one = best_one_vs_one.predict(test_x)
    oneone_score = accuracy_score(test_y, best_pred_one)
    parameters_one =gscv_one.best_params_
    ##one_versus_the_rest
    classifier = OneVsRestClassifier(estimator=SVC())
    parameters = {
        'estimator__C': np.linspace(0.0001, 1000, 10),
        'estimator__kernel': ['rbf', 'linear', 'poly'],
        'estimator__degree': np.arange(1, 6, 1),
        'estimator__gamma': np.linspace(0.0001, 1000, 10)
    }
    gscv_rest = GridSearchCV(estimator = classifier,param_grid = parameters,cv=3,return_train_score=False, verbose=0)
    gscv_rest.fit(train_x, train_y)
    # 最高性能のモデルを取得
    best_one_vs_rest = gscv_rest.best_estimator_
    best_pred_rest = best_one_vs_rest.predict(test_x)
    onerest_score = accuracy_score(test_y, best_pred_rest)
    parameters_rest =gscv_rest.best_params_

    if oneone_score > onerest_score:
        best_score = oneone_score
        best_compare = 'One-versus-one'
        best_pred = best_pred_one
        best_parameters = parameters_one
    else:
        best_score = onerest_score
        best_compare = 'One-versus-the-rest'
        best_pred = best_pred_rest
        best_parameters = parameters_rest

    print('Best score: {}'.format(best_score))
    print('Best parameters: {}'.format(best_parameters))
    print('比較方法:{}'.format(best_compare))
    print('Best pred:{}'.format(best_pred))
    # 混同行列を出力
    #print(confusion_matrix(test_y, best_pred))
    return best_pred




def kNN(train_x, train_y, test_x, test_y):
    k_list = [1,2,3]  # k の数
    weights_list = ['uniform', 'distance']
    ac_score_compare = 0
    for weights in weights_list:
        for k in k_list:
            clf = neighbors.KNeighborsClassifier(k, weights=weights)
            clf.fit(train_x, train_y)
            # 正答率を求める
            pred_y = clf.predict(test_x)
            ac_score = metrics.accuracy_score(pred_y, test_y)
            # print(type(k))
            # print(type(iris_y_test))


            if ac_score_compare == 0:
                ac_score_compare = ac_score
                best_pred = pred_y
                best_k = k
                best_weight = weights
                best_accuracy = ac_score
            elif ac_score_compare < ac_score:
                best_pred = pred_y
                best_k = k
                best_weight = weights
                best_accuracy = ac_score
    print('k={0},weight={1}'.format(best_k, best_weight))
    print('正答率 =', best_accuracy)

    return pred_y


def pCA(x_all, y_all,number,file_name_list):
    # 主成分分析する
    #index_num = 1
    features = x_all
    targets = y_all
    types = np.unique(targets)
    pca = PCA(n_components=2)
    pca.fit(features)
    # 分析結果を元にデータセットを主成分に変換する
    transformed = pca.fit_transform(features)
    # 主成分の寄与率を出力する
    print('各次元の寄与率: {0}'.format(pca.explained_variance_ratio_))
    print('累積寄与率: {0}'.format(sum(pca.explained_variance_ratio_)))

    # 主成分をプロットする
    if number == 0: #厚みの場合

        for label in np.unique(targets): #厚さのみのPCA
            plt.scatter(transformed[targets == label, 0],
                        transformed[targets == label, 1], label='{}mm'.format(label*0.5))
        plt.xlabel('pc1(a.u.)')
        plt.ylabel('pc2(a.u.)')
        plt.legend(loc= 'best')
        #plt.legend(bbox_to_anchor=(1, 0), loc='lower right', borderaxespad=1)
        plt.show()

        for index, (item, file_name) in enumerate(zip(targets, file_name_list)): #ファイル名も表記する。
            if item == 1:
                plt.scatter(transformed[index, 0],
                            transformed[index, 1], marker="${}$".format(file_name), c ='blue')
            if item == 2:
                plt.scatter(transformed[index, 0],
                            transformed[index, 1], marker="${}$".format(file_name), c ='orange')

            elif item == 3:
                plt.scatter(transformed[index, 0],
                            transformed[index, 1], marker="${}$".format(file_name), c ='green')
            elif item == 4:
                plt.scatter(transformed[index, 0],
                            transformed[index, 1], marker="${}$".format(file_name), c ='red')
        plt.xlabel('pc1(a.u.)')
        plt.ylabel('pc2(a.u.)')
        plt.legend(loc='best')
        plt.show()

    else: #糖類の場合
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
        plt.xlabel('pc1(a.u.)')
        plt.ylabel('pc2(a.u.)')
        plt.legend(loc='best')
        plt.show()

        for index, (item, file_name) in enumerate(zip(targets, file_name_list)):  # ファイル名も表記する。
            if item == 1:
                plt.scatter(transformed[index, 0],
                            transformed[index, 1], marker="${}$".format(file_name), c='blue')
            if item == 2:
                plt.scatter(transformed[index, 0],
                            transformed[index, 1], marker="${}$".format(file_name), c='orange')

            elif item == 3:
                plt.scatter(transformed[index, 0],
                            transformed[index, 1], marker="${}$".format(file_name), c='green')
            elif item == 4:
                plt.scatter(transformed[index, 0],
                            transformed[index, 1], marker="${}$".format(file_name), c='red')
            elif item == 5:
                plt.scatter(transformed[index, 0],
                            transformed[index, 1], marker="${}$".format(file_name), c='purple')
            elif item == 6:
                plt.scatter(transformed[index, 0],
                            transformed[index, 1], marker="${}$".format(file_name), c='brown')
        plt.xlabel('pc1(a.u.)')
        plt.ylabel('pc2(a.u.)')
        plt.legend(loc='best')
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

#def RF(train_x,train_y):


#
def smirnov_grubbs(data, alpha):
    x, o = list(data), []
    while True:
        n = len(x)
        t = stats.t.isf(q=(alpha / n) / 2, df=n - 2)
        tau = (n - 1) * t / np.sqrt(n * (n - 2) + n * t * t)
        i_min, i_max = np.argmin(x), np.argmax(x)
        myu, std = np.mean(x), np.std(x, ddof=1)
        i_far = i_max if np.abs(x[i_max] - myu) > np.abs(x[i_min] - myu) else i_min
        tau_far = np.abs((x[i_far] - myu) / std)
        if tau_far < tau: break
        o.append(x.pop(i_far))
    return (np.array(x), np.array(o))
