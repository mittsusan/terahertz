from sklearn.svm import SVC,SVR
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
from sklearn import neighbors
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.decomposition import FastICA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import scipy.stats as stats
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB # ガウシアン
import datetime
import tensorflow as tf
import keras
from keras.layers import Dense,Activation,Dropout,Flatten
from keras.layers.convolutional import Conv1D, UpSampling1D
import keras.backend.tensorflow_backend as KTF
from keras import regularizers
import os

def gaussiannb(train_x, train_y, test_x, test_y):
    gnb = GaussianNB()
    clf = gnb.fit(train_x, train_y)
    y_pred = clf.predict(test_x)
    score = accuracy_score(y_pred, test_y)
    print('score: {}'.format(score))

    return y_pred


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
    if not frequency_list:
        x = np.arange(from_frequency,to_frequency,0.01)
    else:
        x = frequency_list
    print(x)
    print(len(y))
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


def pCA(x_all, y_all, number, file_name_list, type_name_list, concentration_color_type):
    features = x_all
    targets = y_all
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
        plt.xlabel('pc1',fontsize=28)
        plt.ylabel('pc2',fontsize=28)
        plt.legend(loc= 'best',fontsize=16)
        #plt.yticks([-1.0,-0.5,0.0,0.5,1.0])
        plt.tick_params(labelsize=24)
        plt.legend(bbox_to_anchor=(1, 0), loc='lower right', borderaxespad=1)
        plt.show()

        for index, (item, file_name) in enumerate(zip(targets, file_name_list)): #ファイル名も表記する。
            if item == 1:
                plt.scatter(transformed[index, 0],
                            transformed[index, 1], marker="${}$".format(file_name), c ='red')
            elif item == 2:
                plt.scatter(transformed[index, 0],
                            transformed[index, 1], marker="${}$".format(file_name), c ='#%02X%02X%02X' % (0,255,0))

            elif item == 3:
                plt.scatter(transformed[index, 0],
                            transformed[index, 1], marker="${}$".format(file_name), c ='blue')
            elif item == 4:
                plt.scatter(transformed[index, 0],
                            transformed[index, 1], marker="${}$".format(file_name), c ='#%02X%02X%02X' % (255,215,0))

        plt.xlabel('pc1',fontsize=28)
        plt.ylabel('pc2',fontsize=28)
        #plt.legend(loc='best',fontsize=16)
        #plt.xticks([-10, -5, 0, 5, 10])
        #plt.yticks([-10, -5, 0, 5, 10])
        plt.tick_params(labelsize=24)
        plt.show()

        for index, (item, file_name) in enumerate(zip(targets, file_name_list)): #ファイル名も表記する。
            if item == 1:
                plt.scatter(transformed[index, 0],
                            transformed[index, 1], marker="o", c ='red')
            elif item == 2:
                plt.scatter(transformed[index, 0],
                            transformed[index, 1], marker="o", c ='#%02X%02X%02X' % (0,255,0))

            elif item == 3:
                plt.scatter(transformed[index, 0],
                            transformed[index, 1], marker="o", c ='blue')
            elif item == 4:
                plt.scatter(transformed[index, 0],
                            transformed[index, 1], marker="o", c ='#%02X%02X%02X' % (255,215,0))

        plt.xlabel('pc1',fontsize=28)
        plt.ylabel('pc2',fontsize=28)
        #plt.legend(loc='best',fontsize=16)
        #plt.xticks([-10, -5, 0, 5, 10])
        #plt.yticks([-10, -5, 0, 5, 10])
        plt.tick_params(labelsize=24)
        plt.show()

    else: #試薬の場合

        for label, name, concentration_color in zip(np.unique(targets), type_name_list, concentration_color_type):
            plt.scatter(transformed[targets == label, 0],
                        transformed[targets == label, 1], label=name, c = '#%02X%02X%02X' % (concentration_color[0],concentration_color[1],concentration_color[2]))

        plt.xlabel('pc1', fontsize=28)
        plt.ylabel('pc2', fontsize=28)
        #plt.xticks([-2, -1, -0, 0, 1, 2])
        #plt.yticks([-0.75, -0.5, -0.25, 0.00, 0.25, 0.5, 0.75])
        #plt.subplots_adjust(left=0.1, right=0.4, bottom=0.2, top=0.95)
        #plt.legend(loc='best', borderaxespad=0,bbox_to_anchor=(1.05, 1),fontsize=10,ncol=1)
        plt.tick_params(labelsize=24)
        plt.show()

        for index, (item, file_name) in enumerate(zip(targets, file_name_list)):  # ファイル名も表記する。
            plt.scatter(transformed[index, 0],
                        transformed[index, 1], marker="${}$".format(file_name), c='red')

        plt.xlabel('pc1', fontsize=28)
        plt.ylabel('pc2', fontsize=28)
        #plt.xticks([-2, -1, -0, 0, 1, 2])
        #plt.yticks([-0.75, -0.5, -0.25, 0.00, 0.25, 0.5, 0.75])
        #plt.legend(loc='best', fontsize=16)
        plt.tick_params(labelsize=24)
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
    return np.array(x), np.array(o)


def dnn_classification(train_x, train_y, test_x, test_y, class_number, base_dir, from_frequency, to_frequency, frequency_list):
    # conv1 = 30
    nb_epoch = 10000
    nb_batch = 32
    learning_rate = 1e-2
    try:  ##convolutionを使う場合
        conv1

        train_x.resize(train_x.shape[0], train_x.shape[1], 1)
        test_x.resize(test_x.shape[0], test_x.shape[1], 1)
    except:
        pass

    dense1 = 60
    dense2 = 30
    dense3 = 14
    dense4 = class_number
    regularizers_l2_1 = 0
    regularizers_l2_2 = 0
    regularizers_l2_3 = 0

    try:
        model_structure = 'conv{0}relu_{1}relul2{2}_{3}relul2{4}_{5}relul2{6}_{7}softmax'.format(conv1, dense1,
                                                                                                 regularizers_l2_1,
                                                                                                 dense2,
                                                                                                 regularizers_l2_2,
                                                                                                 dense3,
                                                                                                 regularizers_l2_3,
                                                                                                 dense4)
    except:
        model_structure = '{0}relul2{1}_{2}relul2{3}_{4}relul2{5}_{6}softmax'.format(dense1, regularizers_l2_1, dense2,
                                                                                     regularizers_l2_2, dense3,
                                                                                     regularizers_l2_3, dense4)
    f_log = base_dir + '/logs/fit' + 'freq' + str(
        from_frequency) + 'to' + str(to_frequency) + 'num' + str(
        len(frequency_list)) + '/' + model_structure + '_lr' + str(learning_rate) + '/Adam_epoch' + str(
        nb_epoch) + '_batch' + str(nb_batch)
    # print(f_log)
    f_model = base_dir + '/model'  + 'freq' + str(
        from_frequency) + 'to' + str(to_frequency) + 'num' + str(
        len(frequency_list)) + '/' + model_structure + '_lr' + str(learning_rate) + '/Adam_epoch' + str(
        nb_epoch) + '_batch' + str(nb_batch)
    os.makedirs(f_model, exist_ok=True)
    # ニュートラルネットワークで使用するモデル作成
    old_session = KTF.get_session()
    with tf.Graph().as_default():
        session = tf.Session('')
        KTF.set_session(session)
        KTF.set_learning_phase(1)
        model = keras.models.Sequential()
        try:
            model.add(Conv1D(conv1, 4, padding='same', input_shape=(train_x.shape[1:]), activation='relu'))
            model.add(Flatten())
            model.add(Dense(dense1, activation='relu', kernel_regularizer=regularizers.l2(regularizers_l2_1)))
        except:
            model.add(Dense(dense1, activation='relu', kernel_regularizer=regularizers.l2(regularizers_l2_1),
                            input_shape=(train_x.shape[1:])))

        # model.add(Dropout(0.25))
        model.add(Dense(dense2, activation='relu', kernel_regularizer=regularizers.l2(regularizers_l2_2)))
        # model.add(Dropout(0.25))
        model.add(Dense(dense3, activation='relu', kernel_regularizer=regularizers.l2(regularizers_l2_3)))
        model.add(Dense(dense4, activation='softmax'))

        model.summary()
        # optimizer には adam を指定
        adam = keras.optimizers.Adam(lr=learning_rate)

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

        train_y = np.array(train_y)
        test_y = np.array(test_y)
        # print(test_y)
        # print(test_y.shape)
        # print(type(test_y))
        es_cb = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=1000, verbose=0, mode='auto')
        tb_cb = keras.callbacks.TensorBoard(log_dir=f_log, histogram_freq=1)
        # cp_cb = keras.callbacks.ModelCheckpoint(filepath = os.path.join(f_model,'tag_model{epoch:02d}-loss{loss:.2f}-acc{acc:.2f}-vloss{val_loss:.2f}-vacc{val_acc:.2f}.hdf5'), monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
        # cbks = [es_cb, tb_cb, cp_cb]
        cbks = [es_cb, tb_cb]
        history = model.fit(train_x, train_y, batch_size=nb_batch, epochs=nb_epoch,
                            validation_data=(test_x, test_y), callbacks=cbks, verbose=1)
        score = model.evaluate(test_x, test_y, verbose=0)
        print('Test score:', score[0])
        print('Test accuracy:', score[1])
        predict = model.predict(test_x)
        # print('predict:{}'.format(predict))
        print('save the architecture of a model')
        json_string = model.to_json()
        open(os.path.join(f_model, 'tag_model.json'), 'w').write(json_string)
        yaml_string = model.to_yaml()
        open(os.path.join(f_model, 'tag_model.yaml'), 'w').write(yaml_string)
        print('save weights')
        model.save_weights(os.path.join(f_model, 'tag_weights.hdf5'))
    KTF.set_session(old_session)
    best_pred = []
    probability = []
    category = np.arange(1, class_number+1)
    for (i, pre) in enumerate(predict):
        y = pre.argmax()  # preがそれぞれの予測確率で一番高いものを取ってきている。Y_testはone-hotベクトル
        best_pred.append(category[y])
        probability.append(pre[y])
    return best_pred, probability
