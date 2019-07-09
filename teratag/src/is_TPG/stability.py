import pandas as pd
import matplotlib.pyplot as plt

def main(ref,ref2,first_freq,last_freq,x,y,date):

    df_ref = pd.read_table(ref, engine='python', index_col=0)
    df_ref2 = pd.read_table(ref2, engine='python', index_col=0)

    df = df_ref.iloc[:,0]/df_ref2.iloc[:,0]
    df = df[first_freq:last_freq]
    plt.style.use('ggplot')
    df.plot()
    plt.ylim(0, 1.2)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(date)
    plt.show()

    return
if __name__ == '__main__':
    dir_path = '/Users/ryoya/kawaseken/'
    date = '20190701'
    ref_file = '/ref.txt'
    ref2_file = '/ref2.txt'
    x_axis = 'frequency[THz]'
    y_axis = 'transmittance'
    main(dir_path + date + ref_file, dir_path + date + ref2_file, 0.8, 2.6, x_axis, y_axis, date)