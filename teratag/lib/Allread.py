import numpy as np
import pandas as pd
import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
from matplotlib import rcParams
from sklearn import preprocessing
rcParams.update({'figure.autolayout': True})
thickness = ''
sample_init = 0


class allread:
    def __init__(self,method,thickness,type,sample,last_type,last_num):
        #self.df = pd.read_table(file, engine='python')
        #self.file = file
        self.method = method
        self.thickness = thickness
        self.type = type
        self.sample = sample
        self.last_type = last_type
        self.last_num = last_num
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
        #df_polygonal = self.min_max_normalization_TDS(df_polygonal)
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
        self.min_max_normalization()

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
        #self.df.iloc[:,0] = self.df.iloc[:,0]/df_ref.iloc[:,0]
        self.df = self.df[first:last]
        #self.Frequency_trans_reflect_is_TPG_FFT(0) #振幅スペクトルが欲しい場合はnumberを0、位相スペクトルが欲しい時はnumberを1
        self.min_max_normalization()
        #self.graph_Frequency_trans_reflect_is_TPG()
        self.graph_Frequency_trans_reflect_is_TPG_everymm('frequency[THz]',self.method)
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
        plt.xticks([1,1.1,1.2,1.32,1.4,1.5,1.62,1.7,1.8,1.94,2])
        plt.yticks([0,0.2,0.4,0.6,0.8,1.0,1.2])
        plt.ylim(0,1.2)
        plt.gca().xaxis.set_minor_locator(tick.MultipleLocator(0.02))
        plt.gca().yaxis.set_minor_locator(tick.MultipleLocator(0.02))
        plt.xlabel('frequency[THz]')
        plt.ylabel(self.method)
        plt.title(self.thickness)
        thickness = self.thickness
        plt.grid(which='minor')
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
        plt.title('type:'+str(self.type)+'sample:' + str(self.sample))
        #plt.show()

        return

    def graph_Frequency_trans_reflect_is_TPG_everymm(self,x,y):
        global thickness
        global df
        global sample_init


        if sample_init == 0 and self.sample == 1:
            sample_init = 1
            self.df.columns = [self.sample]
            df = self.df
            sample_init = 1
        elif self.sample == 1:
            plt.style.use('ggplot')
            #print('plot')
            df.plot(colormap='tab20')
            plt.xlabel(x)
            plt.ylabel(y)
            plt.title(self.type-1)
            plt.show()
            self.df.columns = [self.sample]
            df = self.df

        else:
            self.df.columns = [self.sample]
            df = df.append(self.df)
        if self.last_type == self.type and self.last_num == self.sample:
            print('lastplot')
            df.plot()
            plt.xlabel(x)
            plt.ylabel(y)
            plt.title(self.type)
            plt.show(colormap='tab20')
            #print(df)
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

    def Frequency_trans_reflect_is_TPG_FFT(self,number):
        list_index = list(self.df.index)
        print(list_index)
        N = len(list_index) #サンプル数
        aliasing = N/2
        dt = round(list_index[1] - list_index[0],2) #サンプリング間隔
        t = np.arange(0, N*dt, dt) # 時間軸
        list_index = list(t)
        #freqList = np.fft.fftfreq(N, d=1.0/fs)  # 周波数軸の値を計算 fsはサンプリング周波数
        a_df = self.df.values
        # ここで一次元にする事でFFT出来るようにする。
        one_dimensional_a_df = np.ravel(a_df)
        print(one_dimensional_a_df)
        #F = np.fft.fft(one_dimensional_a_df)#フーリエ変換
        F = np.fft.ifft(one_dimensional_a_df)#フーリエ逆変換
        #print(F)
        Amp = np.abs(F) #下とおんなじ
        #Amp = [np.sqrt(c.real ** 2 + c.imag ** 2) for c in F]  # 振幅スペクトル
        phaseSpectrum = [np.arctan2(int(c.imag), int(c.real)) for c in F]  # 位相スペクトル
        two_dimentional_Amp = np.reshape(Amp,(len(Amp),1))
        two_dimentional_phase = np.reshape(phaseSpectrum, (len(phaseSpectrum), 1))
        #ここで振幅スペクトルか位相スペクトルかを選ぶ。
        if number == 0:
            self.df = pd.DataFrame(data=two_dimentional_Amp[:int(aliasing)], index=list_index[:int(aliasing)])
        else:
            self.df = pd.DataFrame(data=two_dimentional_phase, index=list_index)
        print(self.df)
        return
