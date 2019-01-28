def svm(train_x, train_y, test_x, test_y):
    param_list = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
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