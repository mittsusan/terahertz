import pandas as pd
import os
from lib import ChangeTransmittance
from lib import ReadFile
import matplotlib.pyplot as plt
import numpy as np
import glob

def main():
    dir_path = '/Users/ryoya/kawaseken/20191016/tag/number2048_average20'
    ref_file = 'ref.txt'
    ref = os.path.join(dir_path,ref_file) #絶対パスにする
    #measurement_file = ['-5db.txt', '-15db.txt', '-25db.txt', 'ems.txt', 'cardboard.txt', 'denim.txt', 'synthetic_leather.txt','leather.txt']
    #measurement_file = ['ems.txt','-5db.txt']
    #measurement_file = ['cardboard.txt','-15db.txt']
    #measurement_file = ['synthetic_leather.txt','-20db.txt']
    #measurement_file =['leather.txt','-50db.txt']
    #measurement_file = ['denim.txt', '-15db.txt']
    #measurement_file = ['1.txt']
    #measurement_file = [f for f in glob.glob(dir_path + 'hakupress2_*.txt')]
    #measurement_file = ['EMS.txt','synthetic.txt','Natural leather.txt']
    #measurement_file = ['PE_porous_1.txt']
    #measurement_file = ['ref_final2.txt']
    #measurement_file = ['ref_final2.txt','PE_porous_1.txt','PE_porous_6.txt']
    #print(measurement_file)
    #measurement_file = ['ref2.txt', 'blank_4.txt', 'blank_6.txt','1_4.txt','1_6.txt','2_4.txt','2_6.txt']
    measurement_file = ['100%.txt', '1.94mm.txt', '1.88mm.txt']
    #measurement_file = ['100%.txt', '1_4.txt', '1_6.txt']
    #measurement_file = ['100%.txt', '2_4.txt', '2_6.txt']
    measurement = [os.path.join(dir_path, file) for file in measurement_file]  # 絶対パスにする
    #print(measurement)

    def transmittance(ref,measurement):
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
        plt.show()
        plt.close()
        return

    def intensity(measurement):
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

        plt.show()
        plt.close()
        return
    def tds_transmittance(ref,measurement):
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
        plt.show()
        plt.close()
        return
    #transmittance(ref,measurement)
    #intensity(measurement)
    tds_transmittance(ref,measurement)
    return

if __name__ == '__main__':
    main()