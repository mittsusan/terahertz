import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

#os.chdir('')
df = pd.read_csv('../../../../sample.csv', header=None, skiprows=2)
a_df = df.values
print(a_df)

#画像の表示
plt.imshow(a_df, cmap = 'gray', vmin = 0, vmax = 255, interpolation = 'none')
# => plt.imshow(img_rgb, interpolation = 'none') と同じ

plt.show()
