import numpy as np
import pandas as pd


# ファイルを読み込む。
# 周波数全て利用
def allread(file):
    ##txtファイルを読み込む。
    df = pd.read_table(file,engine='python')


    x_all = np.empty((0, len(df)), int)
    x_empty = np.empty((0, len(df)), int)

    x_list = []

    for j in df.iloc[:,1]:
        x_list.append(j)

    x_all = np.append(x_all, np.array([x_list]), axis=0)

    return x_all, x_empty