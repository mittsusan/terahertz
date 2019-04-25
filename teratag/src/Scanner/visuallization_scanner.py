import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

#os.chdir('')

df = pd.read_csv('../../../sample.txt', header=None, skiprows=2)
#河上くん用
#df_kawakami = pd.read_table('../../../sample.txt', header=None)
a_df = df.values
#a_df_kawakami = df_kawakami.astype(int).values
#print(a_df_kawakami)

#画像の表示
plt.imshow(a_df, cmap = 'gray', vmin = 0, vmax = 255, interpolation = 'none')
# => plt.imshow(img_rgb, interpolation = 'none') と同じ

plt.show()
