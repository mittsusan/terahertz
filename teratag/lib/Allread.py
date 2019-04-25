import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sklearn import preprocessing
rcParams.update({'figure.autolayout': True})
thickness = ''

class allread:
    def __init__(self,method,thickness):
        #self.df = pd.read_table(file, engine='python')
        #self.file = file
        self.method = method
        self.thickness = thickness

        #self.first_freq = first
        #self.last_freq = last
# ファイルを読み込む。
# 周波数全て利用
    def Time_intensity(self,file):
        self.df = pd.read_table(file, engine='python', names=('時間', '電場', '周波数', '反射率', '位相'))
        self.file = file

        self.graph_Time_intensity_everymm()
        x_list = []

        for j in self.df.iloc[:,1]:
            x_list.append(j)

        x_all = np.array([x_list])

        return x_all

    def Frequency_trans_reflect_TDS(self,file,ref,first,last):
        self.df = pd.read_table(file, engine='python', names=('時間', '電場', '周波数', '反射率', '位相'))
        self.file = file
        self.first_freq = first
        self.last_freq = last
        flag = 0
        x_list = []

        df_ref = pd.read_table(ref, engine='python', names=('時間', '電場', '周波数', '反射率', '位相'))
        #ここで強度を透過率に変化
        self.df.iloc[:,3] = self.df.iloc[:,3]/df_ref.iloc[:,3]
        df_polygonal = self.df.iloc[:, [2, 3]]
        df_polygonal = df_polygonal.set_index('周波数')
        #print(df_polygonal)
        for i,j  in enumerate(df_polygonal.index):
            if flag == 0:
                if j >= first:
                    first_index = i
                    flag = 1
            elif flag == 1:
                if j >= last:
                    last_index = i
                    flag = 2
        df_polygonal = df_polygonal.iloc[first_index:last_index]
        df_polygonal = self.min_max_normalization_TDS(df_polygonal)
        #self.graph_Frequency_trans_reflect()
        self.graph_Frequency_trans_reflect_everymm(df_polygonal)
        #0.2~2THzを見ている。
        for j in df_polygonal.iloc[:,0]:
            x_list.append(j)

        x_all =  np.array([x_list])

        return x_all

    def Frequency_Intencity_is_TPG(self, file, first, last):
        self.df = pd.read_table(file, engine='python', index_col=0)
        self.file = file
        self.first_freq = first
        self.last_freq = last
        x_list = []
        self.df = self.df[first:last]
        print(self.df)
        # self.min_max_normalization()

        self.graph_Frequency_trans_reflect_is_TPG()
        print(self.df)
        for j in self.df.iloc[:, 0]:
            x_list.append(j)

        # [1.12, 1.23, 1.3, 1.36, 1.45, 1.55, 1.6]
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
        #self.Frequency_trans_reflect_is_TPG_FFT()
        self.min_max_normalization()
        #self.graph_Frequency_trans_reflect_is_TPG()
        self.graph_Frequency_trans_reflect_is_TPG_everymm()
        #print(self.df)
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

    def graph_Frequency_trans_reflect_everymm(self,df_polygonal):
        global thickness
        global df
        plt.style.use('ggplot')
        #df_polygonal = self.df.iloc[:, [2, 3]]
        #df_polygonal = df_polygonal.set_index('周波数')
        df_polygonal.columns = [self.file[-5]]
        if thickness != self.thickness:
            df = df_polygonal
        else:
            df = df.append(df_polygonal)

        df.plot()
        plt.xlabel('周波数[THz]')
        plt.ylabel(self.method)
        plt.title(self.thickness)
        thickness = self.thickness
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

    def graph_Time_intensity_everymm(self):
        global thickness
        global df
        plt.style.use('ggplot')
        df_polygonal = self.df.iloc[:, [0, 1]]
        df_polygonal = df_polygonal.set_index('時間')
        df_polygonal.columns = [self.file[-5]]
        if thickness != self.thickness:
            df = df_polygonal
        else:
            df = df.append(df_polygonal)

        df.plot()
        plt.xlabel('時間[ps]')
        plt.ylabel('電場[a.u.]')
        plt.title(self.thickness)
        thickness = self.thickness
        plt.show()
        return

    def graph_Frequency_trans_reflect_is_TPG(self):
        plt.style.use('ggplot')
        df = self.df
        df.columns = [self.method]
        #散布図
        #fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
        df.plot()
        plt.xlabel('周波数[THz]')
        plt.ylabel(self.method)
        plt.title(self.file)
        #plt.show()

        return

    def graph_Frequency_trans_reflect_is_TPG_everymm(self):
        global thickness
        global df
        self.df.columns = [self.file[-5]]
        plt.style.use('ggplot')
        if thickness != self.thickness:
            df = self.df
        else:
            df = df.append(self.df)
        

        #df.columns = [self.method]
        df.plot()
        plt.xlabel('周波数[THz]')
        plt.ylabel(self.method)
        #plt.title(self.file)
        thickness = self.thickness
        #print(thickness)
        plt.show()
        return

    def min_max_normalization(self):
        list_index = list(self.df.index)
        x = self.df.values  # returns a numpy array
        #print(x)
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        #for n in self.drange(self.first_freq,self.last_freq,0.01):
            #list_index.append(n)
        self.df = pd.DataFrame(data=x_scaled,index=list_index)
        return

    def min_max_normalization_TDS(self,df):
        list_index = list(df.index)
        x = df.values  # returns a numpy array
        #print(len(x))
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        #for n in self.drange(self.first_freq,self.last_freq,0.01):
            #list_index.append(n)
        df = pd.DataFrame(data=x_scaled,index=list_index)
        return df

    def drange(self, begin, end, step):
        n = begin
        while n  < end+0.01:
            yield n
            n += step

    def Frequency_trans_reflect_is_TPG_FFT(self):
        list_index = list(self.df.index)
        a_df = self.df.values
        # ここで一次元にする事でFFT出来るようにする。
        one_dimensional_a_df = np.ravel(a_df)
        F = np.fft.fft(one_dimensional_a_df)
        Amp = np.abs(F)
        print(Amp)
        print(len(Amp))
        two_dimentional_Amp = np.reshape(Amp,(len(Amp),1))
        self.df = pd.DataFrame(data=two_dimentional_Amp, index=list_index)
        print(self.df)
        return
