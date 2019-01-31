import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt



class allread:
    def __init__(self,file):
        self.df = pd.read_table(file, engine='python')
        self.file = file
# ファイルを読み込む。
# 周波数全て利用
    def Time_intensity(self):
        x_all = np.empty((0, len(self.df)), int)

        x_list = []

        for j in self.df.iloc[:,1]:
            x_list.append(j)

        x_all = np.array([x_list])

        return x_all

    def Frequency_transmittance(self,ref):
        x_list = []

        df_ref = pd.read_table(ref, engine='python')
        #ここで強度を透過率に変化
        self.df.iloc[:,3] = self.df.iloc[:,3]/df_ref.iloc[:,3]
        self.graph(self.df)
        #0.2~2THzを見ている。
        for j in self.df.iloc[18:165,3]:
            x_list.append(j)

        x_all =  np.array([x_list])

        return x_all

    def graph(self,df):
        #print(matplotlib.rcParams['font.family'])
        plt.style.use('ggplot')
        font = {'family': 'meiryo'}
        matplotlib.rc('font', **font)
        #データの処理
        df = df.iloc[:, [2, 3]]
        df.columns = ['frequency', 'transmittance']
        #散布図
        #fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
        df[18:165].plot(x='frequency', y='transmittance')
        plt.ylabel('transmittance')
        plt.title(self.file)
        plt.show()

        return

