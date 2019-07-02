import openpyxl as px
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import sys
sys.path.append('../../')
from sklearn.cluster import KMeans
import pandas as pd
import csv

book = px.Workbook()
book.save(r'/Users/toshinari/Desktop/python_excell/sample2.xlsx')

y_all = []
flag = 0
l = 0
#使いたいフォルダの選択
path_1 = '/Users/toshinari/Downloads/SVM_train/SVM_train_3'
plt.close()

#med = ['lac_2', 'lac_3', 'lac_4', 'lac_5', 'lac_6']#対象の指定
med = ['lac']
os.chdir(path_1)
x_list = []
M = 0

for w in med:
    for j in sorted(glob.glob("{0}*.txt".format(w))):

        try:
            df = pd.read_csv(j, header = None, names = None,engine = 'python', sep = '\t')
            if M == 0:
                df_1 = df.iloc[:,0]
                #print(df_1)
                M = M+1
                x_list.append(df_1.T)
                df = df.iloc[:,1]
                x_list.append(df.T)
                #print(df)
            else:
                df = df.iloc[:, 1]
                #print(df)
                x_list.append(df.T)

        except FileNotFoundError as e:
            print(e)
    l = l + 1


x_all = pd.DataFrame(x_list)
x_all = x_all.T
print(x_all)
print(x_all.shape)
#x_all = x_all.to_csv('/Users/toshinari/Desktop/python_excell/sample_writer.csv')
x_all.to_excel('/Users/toshinari/Desktop/python_excell/sample_writer.xlsx')