import pandas as pd
import os
class ChangeTransmittance:

    def __init__(self,ref):
        self.ref = ref

    def change_transmittance_list(self,list):
        df_ref = pd.read_table(self.ref, engine='python', index_col=0, header=None)
        flag = True
        for file in list:
            self.df = pd.read_table(file, engine='python', index_col=0, header=None)
            basename = os.path.basename(file)
            root, ext = os.path.splitext(basename)
            self.df.columns = [root]
            self.df.iloc[:, 0] = self.df.iloc[:, 0] / df_ref.iloc[:, 0]
            if flag == True:
                df = self.df
                flag = False

            else:
                df = df.append(self.df,sort=True)
        return df

    def tds_change_transmittance_list(self,list):

        #df_ref_before = pd.read_table(self.ref, engine='python', index_col=0, header=None)
        df_ref = pd.read_table(self.ref, engine='python', names=('Time', 'Intensity', 'Frequency', 'Transmittance', 'Phase'))
        #print(df_ref)
        flag = True
        for file in list:
            self.df = pd.read_table(file, engine='python', names=('Time', 'Intensity', 'Frequency', 'Transmittance', 'Phase'))
            basename = os.path.basename(file)
            root, ext = os.path.splitext(basename)

            self.df.iloc[:, 3] = self.df.iloc[:, 3] / df_ref.iloc[:, 3]
            self.df = self.df.iloc[:, [2, 3]]
            self.df = self.df.set_index('Frequency')
            #print('self.df')
            #print(self.df)
            self.df.columns = [root]
            if flag == True:
                df = self.df
                flag = False
                #print(df)
            else:
                df = df.append(self.df, sort=True)
                #print('グラフ追加')
        return df