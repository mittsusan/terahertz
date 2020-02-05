import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

class allread_scanner:

    def __init__(self,file,title):
        self.file = file
        self.title = title
        x = []
        for i in range(1, 257):
            i = i * (384/256)
            x.append(i)
        self.x = x

    # def func(self,seq):
    #     for z in seq:
    #         try:
    #             yield float(z)
    #         except ValueError:
    #             yield z


    def lightsource_beamshape(self):
        df = pd.read_csv(self.file, engine='python', header=None, skiprows=[0, 1])
        # 通常行ラベルは勝手に番号付けされて、index_col = '0'にすると行ラベルなくなる？？インデックスコラムとは？？
        # 列ラベルは先頭一列目が使われてしまうheader = Noneを使う

        total_average = []
        sum = 0

        for j in range(0, 256):
            #時間軸方向を平均化
            for k in range(0, 512):
                if k == 0:
                    sum = df.iloc[j][k]
                else:
                    sum = sum + df.iloc[j][k]

            average = sum / 512
            # print('平均{}'.format(average))

            total_average.append(average)
        self.plot_graph(total_average)
        #print(len(total_average))

    def lightsource_beamshape_smoothing(self):
        df = pd.read_csv(self.file, engine='python', header=None, skiprows=[0, 1])
        # 通常行ラベルは勝手に番号付けされて、index_col = '0'にすると行ラベルなくなる？？インデックスコラムとは？？
        # 列ラベルは先頭一列目が使われてしまうheader = Noneを使う

        total_average = []
        list = []
        sum = 0
        flag = 0

        for j in range(0, 256):
            #時間軸方向を平均化
            for k in range(0, 512):
                if k == 0:
                    sum = df.iloc[j][k]
                else:
                    sum = sum + df.iloc[j][k]

            average = sum / 512
            # print('平均{}'.format(average))

            total_average.append(average)

        if flag == 0:
            pw_max = max(total_average)
            flag == 1
        else:
            pw_max.append(total_average)
        print(pw_max)

        #平滑化
        for l in range(0,256):
            if l == 0 or l == 255:
                list.append(total_average[l])
            else:
                smoothing_average = (total_average[l-1] + total_average[l] + total_average[l+1])/3
                list.append(smoothing_average)

        self.plot_graph(list)

        return (pw_max)
        #print(len(total_average))

    def plot_graph(self,y):
        plt.plot(self.x,y)
        plt.title('{}'.format(self.title),fontsize=20)
        plt.xlabel('array_distance[mm]',fontsize=19)
        plt.ylabel('beam power[a.u.]',fontsize=19)
        plt.xticks([0,50,100,150,200,250,300,350,384],fontsize=18)
        plt.yticks(fontsize=18)
        plt.show()


        return

    def plot_attenuation(self,y):
        #x_1 = list(range(1,21,1))
        plt.plot(self.x_1, y)
        plt.title('{}_attenuation'.format(self.title))
        plt.xlabel('array_distance[mm]')
        plt.ylabel('beam power[a.u.]')
        plt.xticks(1,20)
        plt.show()
