import pandas as pd
import os
import sys
sys.path.append('../../')
from lib import ChangeTransmittance
from lib import ReadFile
import matplotlib.pyplot as plt
import numpy as np
import glob

def main():
    #ディレクトリ '/dir_path/sensivity/folder_num/file_num.txt' '/dir_path/ref_file'
    dir_path = '/Users/kawaselab/PycharmProjects/20191201/syntheticleather_leather'
    sensivity = 50
    ref_file = 'ref.txt'
    folder_num = 15 #使用する種類の数
    file_num = 5 #使用するファイル数

    ref = os.path.join(dir_path,ref_file) #絶対パスにする

    def transmittance(ref,measurement,i):
        df = ChangeTransmittance(ref).change_transmittance_list(measurement)
        x_axis = 'frequency[THz]'
        y_axis = 'transmittance'
        df.plot(colormap='tab20',legend = False)
        plt.xlim(0.8,1.6)
        plt.ylim(0, 1.2)
        #plt.yscale('log')
        plt.xlabel(x_axis, fontsize=18)
        plt.ylabel(y_axis, fontsize=18)
        plt.tick_params(labelsize=14)
        #plt.xticks(np.arange(1.2, 1.6, 0.04))
        #plt.legend(bbox_to_anchor=(1.01,1), loc=2, borderaxespad=0,fontsize=10)
        plt.title('{}'.format(i))
        plt.show()
        plt.close()
        return

    def intensity(measurement,i):
        df = ReadFile().read_file_list(measurement)
        x_axis = 'frequency[THz]'
        y_axis = 'intensity[mV]'
        df.plot(colormap='tab20', legend=False, grid = False)
        #pointa = 1.0
        #pointb = 1.2
        #pointc = 1.6
        first = 0.8
        last = 1.6
        #plt.xticks([pointa,pointb,pointc,last])
        plt.xlim(first, last)
        plt.ylim(0, 25)
        #plt.vlines(pointa, 0, 25, "red", linestyles='dashed', linewidth=1)
        #plt.vlines(pointb, 0, 25, "red", linestyles='dashed',linewidth = 1)
        #plt.vlines(pointc, 0, 25, "red", linestyles='dashed', linewidth=1)
        #plt.xticks(np.arange(1.0, 1.8, 0.02))
        # plt.yscale('log')
        plt.xlabel(x_axis, fontsize=18)
        plt.ylabel(y_axis, fontsize=18)

        plt.tick_params(labelsize=18)
        plt.legend(fontsize=12)
        plt.title('{}'.format(i))
        plt.show()
        plt.close()
        return
    def tds_transmittance(ref,measurement,i):
        df = ChangeTransmittance(ref).tds_change_transmittance_list(measurement)
        x_axis = 'frequency[THz]'
        y_axis = 'transmittance'
        df.plot(colormap='tab20',legend = False)
        df.plot(colormap='tab20')
        plt.xlim(0.2,0.6)
        plt.ylim(0.6, 1.1)
        #plt.yscale('log')
        plt.xlabel(x_axis, fontsize=18)
        plt.ylabel(y_axis, fontsize=18)
        plt.tick_params(labelsize=14)
        #plt.xticks(np.arange(1.2, 1.6, 0.04))
        #plt.legend(fontsize=12)
        plt.title('{}'.format(i))
        plt.show()
        plt.close()
        return

    for i in range (1,folder_num + 1):
        measurement_file = ['ref2.txt']
        for j in range (1,file_num + 1):
                file_name = '{0}/{1}/{2}.txt'.format(sensivity,i,j)
                measurement_file.append(file_name)

        measurement = [os.path.join(dir_path, file) for file in measurement_file]  # 絶対パスにする

        #transmittance(ref, measurement,i)
        intensity(measurement,i)
        # tds_transmittance(ref,measurement,i)

        if j == file_num:
            measurement_file.clear()


    return

if __name__ == '__main__':
    main()