import pandas as pd
import os
from lib import ChangeTransmittance
from lib import ReadFile
import matplotlib.pyplot as plt
import numpy as np

def main():
    dir_path = '/Users/ryoya/kawaseken/20190724'
    ref_file = 'ref2.txt'
    ref = os.path.join(dir_path,ref_file) #絶対パスにする
    #measurement_file = ['-5db.txt', '-15db.txt', '-25db.txt', 'ems.txt', 'cardboard.txt', 'denim.txt', 'synthetic_leather.txt','leather.txt']
    #measurement_file = ['ems.txt','-5db.txt']
    #measurement_file = ['cardboard.txt','-15db.txt']
    #measurement_file = ['synthetic_leather.txt','-20db.txt']
    #measurement_file =['leather.txt','-50db.txt']
    #measurement_file = ['denim.txt', '-15db.txt']
    measurement_file = ['maltose.txt']
    measurement = [os.path.join(dir_path, file) for file in measurement_file]  # 絶対パスにする

    def transmittance(ref,measurement):
        df = ChangeTransmittance(ref).change_transmittance_list(measurement)
        x_axis = 'frequency[THz]'
        y_axis = 'transmittance'
        df.plot(colormap='tab20',legend = False)
        plt.xlim(0.8,2.6)
        plt.ylim(0, 1.2)
        #plt.yscale('log')
        plt.xlabel(x_axis, fontsize=18)
        plt.ylabel(y_axis, fontsize=18)
        plt.tick_params(labelsize=18)
        plt.legend(fontsize=18)
        plt.show()
        plt.close()
        return

    def intensity(measurement):
        df = ReadFile().read_file_list(measurement)
        x_axis = 'frequency[THz]'
        y_axis = 'intensity[mV]'
        df.plot(colormap='tab20', legend=False, grid=True)
        plt.xlim(1.0, 1.6)
        plt.xticks(np.arange(1.0, 1.8, 0.02))
        # plt.yscale('log')
        plt.xlabel(x_axis, fontsize=18)
        plt.ylabel(y_axis, fontsize=18)

        plt.tick_params(labelsize=4)
        plt.legend(fontsize=18)
        plt.show()
        plt.close()
        return

    #transmittance(ref,measurement)
    intensity(measurement)
    return

if __name__ == '__main__':
    main()