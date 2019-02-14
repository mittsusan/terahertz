import numpy as np

def train_test_split(x_all_2,y_all,train_number):
    train_y = []
    test_y = []
    y_save = 0
    #flagを使って、np.appendのための空の行列をなんとかしている
    flag_train = 0
    flag_test = 0
    number = 0
    for i,j in zip(x_all_2,y_all):
        #ここで正解データが変わった時にスイッチが入る。
        #zipのせいでlistに戻ったので、ndarrayに変更
        i = np.array([i])
        if y_save != j:

            number = 0
            # trainを追加するための処理
            if flag_train == 0:
                train_x = i
                flag_train += 1
            else:
                train_x = np.append(train_x,i,axis=0)
            train_y.append(j)
            number += 1
            y_save = j
        #train_numberの数だけ左からtrainデータを取る。
        elif train_number > number:
            train_x = np.append(train_x,i,axis=0)
            train_y.append(j)
            number += 1
            y_save = j

        else:
            if flag_test == 0:
                test_x = i
                flag_test += 1

            else:
                test_x = np.append(test_x,i,axis=0)
            test_y.append(j)

            y_save = j
    return train_x,train_y,test_x,test_y