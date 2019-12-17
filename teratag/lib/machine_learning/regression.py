import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score
import datetime
import tensorflow as tf
import keras
from keras.layers import Dense,Activation,Dropout,Flatten,LeakyReLU,PReLU
from keras.layers.convolutional import Conv1D, UpSampling1D
import keras.backend.tensorflow_backend as KTF
from keras.wrappers.scikit_learn import KerasRegressor
from keras import regularizers
from sklearn.metrics import r2_score,mean_absolute_error


'''
# データの読み込み
boston = load_boston()

# 訓練データ、テストデータに分割
X, Xtest, y, ytest = train_test_split(boston['data'], boston['target'], test_size=0.2, random_state=114514)
# 6:2:2に分割にするため、訓練データのうちの後ろ1/4を交差検証データとする
# 交差検証データのジェネレーター
def gen_cv():
    m_train = np.floor(len(y)*0.75).astype(int)#このキャストをintにしないと後にハマる
    train_indices = np.arange(m_train)
    test_indices = np.arange(m_train, len(y))
    yield (train_indices, test_indices)
# (それぞれ303 101 102 = サンプル合計は506)
print("リッジ回帰")
print()
print("訓練データ、交差検証データ、テストデータの数 = ", end="")
print(len(next(gen_cv())[0]), len(next(gen_cv())[1]), len(ytest) )
print()

# 訓練データを基準に標準化（平均、標準偏差で標準化）
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)
# テストデータも標準化
Xtest_norm = scaler.transform(Xtest)

'''
###ここから分岐
def ridge(train_x, train_y, test_x, test_y):
    # ハイパーパラメータのチューニング
    params = {"alpha":np.logspace(-2, 4, 24)}
    gridsearch = GridSearchCV(Ridge(), params, cv=3, scoring="r2", return_train_score=True)
    gridsearch.fit(train_x, train_y)
    print("αのチューニング")
    print("最適なパラメーター =", gridsearch.best_params_, "精度 =", gridsearch.best_score_)
    print()

    # 検証曲線
    plt.semilogx(params["alpha"], gridsearch.cv_results_["mean_train_score"], label="Training")
    plt.semilogx(params["alpha"], gridsearch.cv_results_["mean_test_score"], label="Cross Validation")
    plt.xlabel("alpha")
    plt.ylabel("R2 Score")
    plt.title("Validation curve / Linear Regression")
    plt.legend()
    plt.show()

    # チューニングしたαでフィット
    regr = Ridge(alpha=gridsearch.best_params_["alpha"])
    regr.fit(train_x, train_y)
    print("切片と係数")
    print(regr.intercept_)
    print(regr.coef_)
    print()
    # テストデータの精度を計算
    print("テストデータにフィット")
    print("テストデータの精度 =", regr.score(test_x, test_y))
    print()

def svr_linear(train_x, train_y, test_x, test_y):

    # ハイパーパラメータのチューニング
    # 計算に時間がかかるのである程度パラメーターを絞っておいた
    # （1e-2～1e4まで12×12でやって最適値が'C': 0.123, 'epsilon': 1.520）
    params_cnt = 20
    params = {"C":np.logspace(0,1,params_cnt), "epsilon":np.logspace(-1,1,params_cnt)}
    gridsearch = GridSearchCV(SVR(kernel="linear"), params, cv=3, scoring="r2", return_train_score=True)
    gridsearch.fit(train_x, train_y)
    print("C, εのチューニング")
    print("最適なパラメーター =", gridsearch.best_params_)
    print("精度 =", gridsearch.best_score_)
    print()

    # チューニングしたハイパーパラメーターをフィット
    regr = SVR(kernel="linear", C=gridsearch.best_params_["C"], epsilon=gridsearch.best_params_["epsilon"])
    regr.fit(train_x, train_y)
    print("切片と係数")
    print(regr.intercept_)
    print(regr.coef_)
    print()
    # テストデータの精度を計算
    print("テストデータにフィット")
    print("テストデータの精度 =", regr.score(test_x, test_y))
    print()


def svr_rbf(train_x, train_y, test_x, test_y):

    # ハイパーパラメータのチューニング
    params_cnt = 20
    params = {"C":np.logspace(0,2,params_cnt), "epsilon":np.logspace(-1,1,params_cnt)}
    gridsearch = GridSearchCV(SVR(), params, cv=3, scoring="r2", return_train_score=True)
    gridsearch.fit(train_x, train_y)
    print("C, εのチューニング")
    print("最適なパラメーター =", gridsearch.best_params_)
    print("精度 =", gridsearch.best_score_)
    print()

    # 検証曲線
    plt_x, plt_y = np.meshgrid(params["C"], params["epsilon"])
    fig = plt.figure(figsize=(8,8))
    fig.subplots_adjust(hspace = 0.3)
    for i in range(2):
        if i==0:
            plt_z = np.array(gridsearch.cv_results_["mean_train_score"]).reshape(params_cnt, params_cnt, order="F")
            title = "Train"
        else:
            plt_z = np.array(gridsearch.cv_results_["mean_test_score"]).reshape(params_cnt, params_cnt, order="F")
            title = "Cross Validation"
        ax = fig.add_subplot(2, 1, i+1)
        CS = ax.contour(plt_x, plt_y, plt_z, levels=[0.6, 0.65, 0.7, 0.75, 0.8, 0.85])
        ax.clabel(CS, CS.levels, inline=True)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("C")
        ax.set_ylabel("epsilon")
        ax.set_title(title)
    plt.suptitle("Validation curve / Gaussian SVR")
    plt.show()


    # チューニングしたC,εでフィット
    regr = SVR(C=gridsearch.best_params_["C"], epsilon=gridsearch.best_params_["epsilon"])
    regr.fit(train_x, train_y)
    # テストデータの精度を計算
    print("テストデータにフィット")
    print("テストデータの精度 =", regr.score(test_x, test_y))
    print()

def ridge_multi(train_x, train_y, test_x, test_y):
    # ハイパーパラメータのチューニング
    params = {"estimator__alpha":np.logspace(-2, 4, 24)}
    gridsearch = GridSearchCV(MultiOutputRegressor(Ridge()), params, cv=3, scoring="r2", return_train_score=True)
    gridsearch.fit(train_x, train_y)
    print("αのチューニング")
    print("最適なパラメーター =", gridsearch.best_params_, "精度 =", gridsearch.best_score_)
    print()

    # チューニングしたαでフィット
    regr = MultiOutputRegressor(Ridge(alpha=gridsearch.best_params_["estimator__alpha"]))
    regr.fit(train_x, train_y)
    # テストデータの精度を計算
    print("テストデータの精度 =", regr.score(test_x, test_y))
    print()

def svr_linear_multi(train_x, train_y, test_x, test_y):

    # ハイパーパラメータのチューニング
    # 計算に時間がかかるのである程度パラメーターを絞っておいた
    # （1e-2～1e4まで12×12でやって最適値が'C': 0.123, 'epsilon': 1.520）
    params_cnt = 20
    params = {"estimator__C":np.logspace(0,1,params_cnt), "estimator__epsilon":np.logspace(-1,1,params_cnt)}
    gridsearch = GridSearchCV(MultiOutputRegressor(SVR(kernel="linear")), params, cv=3, scoring="r2", return_train_score=True)
    gridsearch.fit(train_x, train_y)
    print("C, εのチューニング")
    print("最適なパラメーター =", gridsearch.best_params_)
    print("精度 =", gridsearch.best_score_)
    print()

    # チューニングしたハイパーパラメーターをフィット
    regr = MultiOutputRegressor(SVR(kernel="linear", C=gridsearch.best_params_["estimator__C"], epsilon=gridsearch.best_params_["estimator__epsilon"]))
    regr.fit(train_x, train_y)
    # テストデータの精度を計算
    print("テストデータの精度 =", regr.score(test_x, test_y))
    print()


def svr_rbf_multi(train_x, train_y, test_x, test_y):

    # ハイパーパラメータのチューニング
    params_cnt = 20
    params = {"estimator__C":np.logspace(0,2,params_cnt), "estimator__epsilon":np.logspace(-1,1,params_cnt)}
    gridsearch = GridSearchCV(MultiOutputRegressor(SVR()), params, cv=3, scoring="r2", return_train_score=True)
    gridsearch.fit(train_x, train_y)
    print("C, εのチューニング")
    print("最適なパラメーター =", gridsearch.best_params_)
    print("精度 =", gridsearch.best_score_)
    print()

    # チューニングしたC,εでフィット
    regr = MultiOutputRegressor(SVR(C=gridsearch.best_params_["estimator__C"], epsilon=gridsearch.best_params_["estimator__epsilon"]))
    regr.fit(train_x, train_y)
    # テストデータの精度を計算
    print("テストデータの精度 =", regr.score(test_x, test_y))
    print()

def randomforest_regression(train_x, train_y, test_x, test_y, from_frequency, to_frequency, frequency_list):
    # use a full grid over all parameters
    # param_grid = {"n_estimators": np.arange(50,300,10)}
    #
    # forest_grid = GridSearchCV(estimator=RandomForestRegressor(random_state=0,bootstrap = True),
    #                            param_grid=param_grid,
    #                            cv=3)
    # forest_grid_best = forest_grid.best_estimator_  # best estimator
    # print("Best Model Parameter: ", forest_grid.best_params_)
    # best_pred = forest_grid_best.predict(test_x)
    forest_grid = RandomForestRegressor(n_estimators=120,random_state=0,bootstrap=True)
    forest_grid.fit(train_x, train_y)  # fit
    best_pred = forest_grid.predict(test_x)
    print("決定係数:", forest_grid.score(test_x, test_y))
    print('Mean absolute error 誤差率:',mean_absolute_error(test_y,best_pred))
    # 特徴量の重要度
    feature_importances = forest_grid.feature_importances_
    plt.figure(figsize=(10, 5))
    y = feature_importances
    if not frequency_list:
        x = np.arange(from_frequency,to_frequency+0.01,0.01)
    else:
        x = frequency_list
    # print(len(x))
    # print(len(y))
    plt.bar(x, y, width = 0.005, align="center")
    plt.xlabel('Frequency (THz)')
    plt.ylabel('Feature importance')
    plt.show()

    return best_pred

def dnn(train_x, train_y, test_x, test_y):
    model = MLPRegressor(hidden_layer_sizes=(100,100,100,100,),random_state=0,max_iter=5000)
    model.fit(train_x, train_y)
    print("テストデータの精度 =", model.score(test_x, test_y))



def keras_dnn(train_x, train_y, test_x, test_y,from_frequency, to_frequency, frequency_list, class_number,
              shielding_material, nb_epoch, nb_batch, learning_rate,dense1, dense2, dense3, dense4, regularizers_l2_1, regularizers_l2_2, regularizers_l2_3):
    # conv1 = 30
    try:  ##convolutionを使う場合
        conv1
        print(train_x.shape[1:])
        train_x.resize(train_x.shape[0], train_x.shape[1], 1)
        test_x.resize(test_x.shape[0], test_x.shape[1], 1)
    except:
        pass

    # print(train_x.shape)
    # print(train_y)

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
    f_log = 'User/ryoya/kawaseken/teratag/logs/fit' + shielding_material + 'freq' + str(
        from_frequency) + 'to' + str(to_frequency) + 'num' + str(
        len(frequency_list)) + '/' + model_structure + '_lr' + str(learning_rate) + '/Adam_epoch' + str(
        nb_epoch) + '_batch' + str(nb_batch)
    # print(f_log)
    f_model = 'User/ryoya/kawaseken/teratag/model' + shielding_material + 'freq' + str(
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
        model.add(Dense(dense4, activation='linear'))
        #model.add(Dense(dense4, activation=LeakyReLU()))
        model.summary()
        # optimizer には adam を指定
        adam = keras.optimizers.Adam(lr=learning_rate)

        model.compile(loss='mse', optimizer='adam')
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
        predict = model.predict(test_x)
        print('元データ:{}'.format(test_y))
        print('予想データ:{}'.format(predict))
        print('決定係数:',r2_score(test_y,predict))
        print('Mean absolute error 誤差率:',mean_absolute_error(test_y, predict))
        print('save the architecture of a model')
        json_string = model.to_json()
        open(os.path.join(f_model, 'tag_model.json'), 'w').write(json_string)
        yaml_string = model.to_yaml()
        open(os.path.join(f_model, 'tag_model.yaml'), 'w').write(yaml_string)
        print('save weights')
        model.save_weights(os.path.join(f_model, 'tag_weights.hdf5'))
    KTF.set_session(old_session)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['loss', 'val_loss'], loc='lower right')
    plt.show()

def keras_dnn_predict(train_x, train_y, test_x, test_y,from_frequency, to_frequency, frequency_list, class_number, shielding_material, nb_epoch, nb_batch, learning_rate,model_structure):
    ##保存したモデルと重みを使用して、予測。
    from keras.models import model_from_json
    import keras.backend.tensorflow_backend as KTF
    import tensorflow as tf
    import keras
    f_log = 'User/kawaseken/teratag/logs/fit' + shielding_material + 'freq' + str(
        from_frequency) + 'to' + str(to_frequency) + 'num' + str(
        len(frequency_list)) + '/' + model_structure + '_lr' + str(learning_rate) + '/Adam_epoch' + str(
        nb_epoch) + '_batch' + str(nb_batch)
    # print(f_log)
    f_model = 'User/kawaseken/teratag/model' + shielding_material + 'freq' + str(
        from_frequency) + 'to' + str(to_frequency) + 'num' + str(
        len(frequency_list)) + '/' + model_structure + '_lr' + str(learning_rate) + '/Adam_epoch' + str(
        nb_epoch) + '_batch' + str(nb_batch)
    old_session = KTF.get_session()
    # ニュートラルネットワークで使用するモデル作成
    model_filename = 'tag_model.json'
    weights_filename = 'tag_weights.hdf5'
    with tf.Graph().as_default():
        session = tf.Session('')
        KTF.set_session(session)

        json_string = open(os.path.join(f_model, model_filename)).read()
        model = model_from_json(json_string)

        model.summary()
        adam = keras.optimizers.Adam(lr=learning_rate)
        model.compile(loss='mse', optimizer='adam')

        model.load_weights(os.path.join(f_model, weights_filename))

        cbks = []
        predict = model.predict(test_x)
        print('元データ:{}'.format(test_y))
        print('予想データ:{}'.format(predict))
    KTF.set_session(old_session)