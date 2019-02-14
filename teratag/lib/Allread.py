import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn import preprocessing
rcParams.update({'figure.autolayout': True})


class allread:
    def __init__(self,method):
        #self.df = pd.read_table(file, engine='python')
        #self.file = file
        self.method = method
        #self.first_freq = first
        #self.last_freq = last
# ファイルを読み込む。
# 周波数全て利用
    def Time_intensity(self,file):
        self.df = pd.read_table(file, engine='python')
        self.file = file

        self.graph_Time_intensity()
        x_list = []

        for j in self.df.iloc[:,1]:
            x_list.append(j)

        x_all = np.array([x_list])

        return x_all

    def Frequency_trans_reflect_TDS(self,file,ref,first,last):
        self.df = pd.read_table(file, engine='python')
        self.file = file
        self.first_freq = first
        self.last_freq = last
        x_list = []

        df_ref = pd.read_table(ref, engine='python')
        #ここで強度を透過率に変化
        self.df.iloc[:,3] = self.df.iloc[:,3]/df_ref.iloc[:,3]
        self.graph_Frequency_trans_reflect()
        #0.2~2THzを見ている。
        for j in self.df.iloc[18:165,3]:
            x_list.append(j)

        x_all =  np.array([x_list])

        return x_all

    def Frequency_Intencity_is_TPG(self,file,first,last):
        self.df = pd.read_table(file, engine='python', index_col=0)
        self.file = file
        self.first_freq = first
        self.last_freq = last
        x_list = []
        self.df = self.df[first:last]
        print(self.df)
        #self.min_max_normalization()

        self.graph_Frequency_trans_reflect_is_TPG()
        print(self.df)
        for j in self.df.iloc[:, 0]:
            x_list.append(j)

        #[1.12, 1.23, 1.3, 1.36, 1.45, 1.55, 1.6]
        x_all = np.array([x_list])

        return x_all

    def Frequency_trans_reflect_is_TPG(self,file,ref,first,last):
        self.df = pd.read_table(file, engine='python',index_col=0)
        self.file = file
        self.first_freq = first
        self.last_freq = last
        x_list = []

        df_ref = pd.read_table(ref, engine='python',index_col=0)
        #ここで強度を透過率に変化
        self.df.iloc[:,0] = self.df.iloc[:,0]/df_ref.iloc[:,0]
        self.df = self.df[first:last]

        self.min_max_normalization()
        self.graph_Frequency_trans_reflect_is_TPG()
        print(self.df)
        for j in self.df.iloc[:,0]:
            x_list.append(j)

        x_all =  np.array([x_list])

        return x_all

    def graph_Frequency_trans_reflect(self):
        #print(matplotlib.rcParams['font.family'])
        plt.style.use('ggplot')
        #font = {'family': 'meiryo'}
        #matplotlib.rc('font', **font)
        #データの処理
        df = self.df.iloc[:, [2, 3]]
        df.columns = ['frequency', self.method]
        #散布図
        #fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
        df[18:165].plot(x='frequency', y=self.method)
        plt.ylabel(self.df)
        plt.title(self.file)
        plt.show()

        return

    def graph_Time_intensity(self):
        #print(matplotlib.rcParams['font.family'])
        plt.style.use('ggplot')
        #font = {'family': 'meiryo'}
        #matplotlib.rc('font', **font)
        #データの処理
        df = self.df.iloc[:, [0, 1]]
        df.columns = ['time', 'intensity']
        #散布図
        #fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
        df.plot(x='time', y='intensity')
        plt.ylabel('intensity')
        plt.title(self.file)
        plt.show()

        return

    def graph_Frequency_trans_reflect_is_TPG(self):

        plt.style.use('ggplot')
        df = self.df
        df.columns = [self.method]
        matplotlib.rcParams['font.family'] = 'AppleGothic'
        #散布図
        #fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
        df.plot()
        plt.xlabel('周波数[THz]')
        plt.ylabel(self.method)
        plt.title(self.file)
        plt.show()

        return

    def min_max_normalization(self):
        list_index = []
        x = self.df.values  # returns a numpy array
        #print(x)
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        for n in self.drange(self.first_freq,self.last_freq,0.01):
            list_index.append(n)
        self.df = pd.DataFrame(data=x_scaled,index=list_index)
        return

    def drange(self, begin, end, step):
        n = begin
        while n  <= end:
            yield n
            n += step
