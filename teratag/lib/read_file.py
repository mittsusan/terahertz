import pandas as pd
import os

class ReadFile:

    def __init__(self):
        pass

    def read_file_list(self,list):
        flag = True
        for file in list:
            self.df = pd.read_table(file, engine='python', index_col=0, header=None)
            basename = os.path.basename(file)
            root, ext = os.path.splitext(basename)
            self.df.columns = [root]
            if flag == True:
                df = self.df
                flag = False

            else:
                df = df.append(self.df,sort=True)
        return df