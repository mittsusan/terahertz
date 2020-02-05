import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('../../')
from lib.Allread_scanner import allread_scanner

dir_path = '/Users/kawaselab/PycharmProjects/scanner/20200128/cylindrical'


for i in range(0,31):
    #i = i * 5
    title = 'cardboard_{}'.format(i)
    file_name = '{}_raw.csv'.format(i)
    file = os.path.join(dir_path,file_name)
    #col = ['{0}'.format(j) for j in range(512)]
    try:
        pw_max = allread_scanner(file,title).lightsource_beamshape_smoothing()
    except FileNotFoundError as e:
        print(e)
allread_scanner(file,title).plot_attenuation(pw_max)