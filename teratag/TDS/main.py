
import numpy as np
from TDS.TDS_read import allread

#refrenceデータを読み込む。
#ここでの実行のは空の行列(x_all)を取ってくるため。
x,x_all = allread('/Users/ryoya/kawaseken/20190123/2019_0123_ref_1.txt')
y_all = []
#mainデータを読み込む。
for i in range(1,4):
    for j in range(1,6):
        try:
            x,x_empty = allread('/Users/ryoya/kawaseken/20190123/2019_0123_{0}mm_{1}.txt'.format(i,j))
            x_all = np.append(x_all, x, axis=0)
            y_all.append(i)
        except FileNotFoundError as e:
            print(e)
        #print(x)

print(x_all)
print(y_all)