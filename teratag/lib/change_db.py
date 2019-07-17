import numpy as np
import pandas as pd

def change_db(df):
    values = 10 * np.log10(df.values)
    #values[np.isnan(values)] = 0 #nanの場合0に変換
    df.iloc[:,0] = values
    return df