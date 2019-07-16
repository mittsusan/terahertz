import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import glob
import os
from sklearn.model_selection import train_test_split, GridSearchCV

# Pandas のデータフレームとして表示

#データの定義
sample_list = ['glu','mal','lac']
first = 1.1
last = 1.8
x_list = []
y_list = []
l=0
j = 0
f_num = []


#以下教師データのラベルづけ
for w in sample_list:
    path_1 = os.chdir('/Users/toshinari/Downloads/SVM_file/ALL_RESULT/ALL_file/Trans/{}'.format(w))

    file_list = sorted(glob.glob('/Users/toshinari/Downloads/SVM_file/ALL_RESULT/ALL_file/Trans/{}/*'.format(w)))
    #print(file_list)
    file_number = len(file_list)
    f_num.append(file_number)
    print(file_number)
    #a = np.empty((121,file_number))
    for k in range(0,file_number):
        df = pd.read_csv(file_list[k], engine='python', header=None, index_col=0, sep=',')
        df = df[first:last]#特定の周波数範囲の抜き取り
        #print(df)
        #df = Max_Min(df)#正規化
        #print(df.iloc[:,0])
        #ここまで欲しいところを抜き出している過程
        df_np = df.values
        #print(df_np)
        if k ==0:
            x_all = df_np
            all_f = df
            #j = j + 1
        else:
            x_all = np.append(x_all,df_np,axis = 1)
            all_f[k+1] = df[1]
        y_list.append(l)
    #x_all = np.array([x_list])
    if l==0:
        X_all = x_all
        All_f = all_f
    else:
        X_all = np.append(X_all,x_all, axis = 1)
        All_f = pd.concat([All_f,all_f],axis = 1)
    l = l+1
y_all = np.array(y_list)

print(All_f.shape)
print(f_num)
#学習データとテストデータに分割
x_train, x_test, y_train, y_test = train_test_split(X_all.T, y_all, test_size=0.20)

# 学習
clf = RandomForestClassifier(n_estimators=20, random_state=42)
clf.fit(x_train, y_train)

#予測データ作成
y_predict = clf.predict(x_test)

#正解率
print('\n正答率')
print(accuracy_score(y_test, y_predict))
print(y_test)
print(y_predict)

#gs = GridSearchCV(RandomForestClassifier(),search_params,cv = 3, verbose True,n_jobs=-1)


#特徴量の重要度
feature = clf.feature_importances_
f = pd.DataFrame({'number': range(0, len(feature)),
             'feature': feature[:]})



f2 = f.sort_values('feature',ascending=False)
f3 = f2.ix[:, 'number']




df = pd.read_csv(file_list[k], engine='python', header=None, sep=',')
#print(int((first-0.8)*100+1))
#print(int((last-0.8)*100+2))
df = df[int((first-0.8)*100+1):int((last-0.8)*100+2)]#周波数の値を位置に直したかった。なんか他に方法あるかもやけどめんどかったからこうしてる。改善してくれ。
df_np_2 = df.values
df_np2 = df_np_2[:,0]
X_all2 = np.insert(X_all.T, 0, df_np2, axis = 0)#多次元配列に一次元配列を追加する場合はappendやstackでは無理。insertを使う
#print(X_all2)


#print('Feature Importances:')
#for i, feat in enumerate(iris['feature_names']):
 #   print('\t{0:20s} : {1:>.6f}'.format(feat, fti[i]))

#firstとlast、つまり周波数範囲に応じた１０刻みからなるリスト
i = int((last-first)*10)
freq_list = []
freq_list2 = []
for p in range(0,i+1):
    p = int(p)
    freq_list.append(10*p)
    freq_list2.append(round ( (1.1+(p/10)),3))


print(freq_list2)
#print(feature.shape)
#for i in range(0,len(feature)):
    #print('\t{0}:{1}'.format(X_all2[0,i], feature[i]))
fig = plt.figure()
plt.title('Fiture Impotance')
left = X_all2[0,:]
#print(X_all2[0,:])
X_list = X_all2[0,:].tolist()
X_list_str = [str(n) for n in X_list]
print(X_list_str)
height = feature
ax = fig.add_subplot(1,1,1)
ax.bar(range(len(feature)),height)
#ax.set_xticks([0,10,20,30,40,50,60,70])
#ax.set_xticklabels([1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8], rotation=30)
ax.set_xticks(freq_list)
ax.set_xticklabels(freq_list2, rotation=30)

#グラフの表示
plt.show()


'''
plt.title('Fiture Impotance')
left = X_all2[0,:]
#print(X_all2[0,:])
X_list = X_all2[0,:].tolist()
X_list_str = [str(n) for n in X_list]
print(X_list_str)
height = feature
ax = plt.bar(range(len(feature)),height)
plt.set_xticklabels(X_all2[0,:], rotation=90)
#plt.bar(X_all2[0,:].T,height)
plt.xticks()

plt.show()

'''
'''
#特徴量の重要度を上から順に出力する
f = pd.DataFrame({'number': range(0, len(feature)),
             'feature': feature[:]})
f2 = f.sort_values('feature',ascending=False)
f3 = f2.ix[:, 'number']

#特徴量の名前
label = df.columns[0:]

#特徴量の重要度順（降順）
indices = np.argsort(feature)[::-1]
'''
'''
for i in range(len(feature)):
    print(str(i + 1)) + "   " + str(label[indices[i]]) + "   " + str(feature[indices[i]])
'''
'''
plt.title('Feature Importance')
plt.bar(range(len(feature)),feature[indices], color='lightblue', align='center')
plt.xticks(range(len(feature)), label[indices], rotation=90)
plt.xlim([-1, len(feature)])
plt.tight_layout()
plt.show()

'''