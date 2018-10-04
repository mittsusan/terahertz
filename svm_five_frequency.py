from sklearn.datasets import load_digits
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import math

#
# 決定境界プロット関数
#
def plot_decision_regions(x, y, model, resolution=0.01):

    ## 今回は被説明変数が3クラスのため散布図のマーカータイプと3種類の色を用意
    ## クラスの種類数に応じて拡張していくのが良いでしょう
    markers = ('s', 'x', 'o')
    cmap = ListedColormap(('red', 'blue', 'green'))

    ## 2変数の入力データの最小値から最大値まで引数resolutionの幅でメッシュを描く
    x1_min, x1_max = x[:, 0].min()-1, x[:, 0].max()+1
    x2_min, x2_max = x[:, 1].min()-1, x[:, 1].max()+1
    x1_mesh, x2_mesh = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                                   np.arange(x2_min, x2_max, resolution))

    ## メッシュデータ全部を学習モデルで分類
    z = model.predict(np.array([x1_mesh.ravel(), x2_mesh.ravel()]).T)
    z = z.reshape(x1_mesh.shape)

    ## メッシュデータと分離クラスを使って決定境界を描いている
    plt.contourf(x1_mesh, x2_mesh, z, alpha=0.4, cmap=cmap)
    plt.xlim(x1_mesh.min(), x1_mesh.max())
    plt.ylim(x2_mesh.min(), x2_mesh.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=x[y == cl, 0],
                    y=x[y == cl, 1],
                    alpha=0.6,
                    c=cmap(idx),
                    edgecolors='black',
                    marker=markers[idx],
                    label=cl)


arr = np.empty((0,5), int)
thickness = 1.0
sample = 1
frequency = [1.275,1.369,1.496,1.629,1.737]
param_list = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
Y = []
best_score = 0
best_parameters = {}
kernel = 'rbf'
for i in range(1,4):
    #print('厚さ:{}'.format(thickness))
    for j in range(1,23):
        #print('sample:{}'.format(j))
        f = open('/Users/ryoya/kawaseken/多波長測定8_3/透過率/{0}mm/10点 {1}.txt'.format(thickness,j))
        data1 = f.read()  # ファイル終端まで全て読んだデータを返す
        f.close()
        lines1 = data1.split('\n') # 改行で区切る(改行文字そのものは戻り値のデータには含まれない)
        for line in lines1:
            line2 = line.split('\t')
            X = []
            for k, transmittance in enumerate(line2):
                    #X.append(frequency[k])
                    X.append(transmittance)
                    #print('X_list:{}'.format(X))
            #print(X)
            if not transmittance == '':
                arr = np.append(arr, np.array([X]), axis=0)
                Y.append(sample)
    thickness += 0.5
    sample += 1
#digits = load_digits()
print('X:{}'.format(arr))
print('Y:{}'.format(Y))
print('X.shape:{}'.format(arr.shape))
print('Y.shape:{}'.format(len(Y)))
train_x, test_x, train_y, test_y  = train_test_split(arr, Y,test_size=0.3,random_state=0) #defaultでtestサイズ0.25
#print('digit.data:{0},digit.target:{1}'.format(digits.data,digits.target))
for gamma in param_list:#グリッドサーチをしてハイパーパラメータ探索
    for C in param_list:
        estimator = SVC(gamma=gamma, kernel=kernel,C=C)
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
        else:
            score = oneone_score
            hikaku = 'One-versus-one'
        """
        svm.fit(train_x, train_y)
        score = svm.score(test_x, test_y)#test_xのラベル付けをして、test_yとの比較をしてスコア計算
        """
        # 最も良いスコアのパラメータとスコアを更新
        if score > best_score:
            best_hikaku = hikaku
            best_score = score
            best_parameters = {'gamma' : gamma, 'C' : C}

print('Best score: {}'.format(best_score))
print('Best parameters: {}'.format(best_parameters))
print('比較方法:{}'.format(best_hikaku))
"""
C = 1.

gamma  = 0.01
estimator = SVC(C=C, kernel=kernel, gamma=gamma)
classifier = OneVsRestClassifier(estimator)
classifier.fit(train_x, train_y)
pred_y = classifier.predict(test_x)
classifier2 = SVC(C=C, kernel=kernel, gamma=gamma)
classifier2.fit(train_x, train_y)
pred_y2 = classifier2.predict(test_x)
print ('One-versus-the-rest: {:.5f}'.format(accuracy_score(test_y, pred_y)))
print ('One-versus-one: {:.5f}'.format(accuracy_score(test_y, pred_y2)))
"""
