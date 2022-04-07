'''
对于上面的数据，把他放进了一个表格进行分析，发现并不能解决这个问题
无论是使用下面这个系数，还是重新进行拟合之后的系数，都没办法复现第一页计算结果中的DeltaTemp数据，总归都是由比较大的差异
'''

import os, pandas , time, xlrd, openpyxl,random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pywt #python wavelet transmission
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import linear_model
def remove_minusV(originaldata):
    #删除电压小于零的部分，有一些可能是无用数据
    newdata = originaldata.drop(originaldata[originaldata['电解电压']<0.5].index)
    newdata.index = np.arange(len(newdata))
    return newdata
def retrieve_temp(DeltaTempPredicted,InitialTemp):
    TempRetrieved = np.empty(len(DeltaTempPredicted))
    TempRetrieved[0] = float(InitialTemp)
    for i in np.arange(1,len(TempRetrieved)):
        TempRetrieved[i] = TempRetrieved[i-1] + DeltaTempPredicted[i]
    return TempRetrieved

def compare_DeltaandTemp(DataSet,PredDeltaTemp,file,plt):
    ax1 = plt.gca()
    ax1.set_ylabel(r'$Temperature \quad  \circ_C$')
    ax1.set_xlabel('Time')
    ax1.plot(np.arange(len(DataSet)), DataSet['OriTemp'])
    ax1.plot(np.arange(len(DataSet)), retrieve_temp(PredDeltaTemp, DataSet['OriTemp'][0]),'r')
    ax1.legend(['Original Data', 'Predicted Temp'], loc=1)

    ax2 = ax1.twinx()
    ax2.set_ylabel(r'$\Delta \quad Temp \circ_C$')
    ax2.plot(np.arange(len(DataSet)), DataSet['DeltaTemp'])
    ax2.plot(np.arange(len(DataSet)), PredDeltaTemp)
    ax2.legend(['Original Data', r'$Predicted \quad  \Delta T$'], loc=4)
    plt.title(file)


file = "DataSample-1228.csv"
df = pandas.read_csv(file)

clf = LinearRegression(fit_intercept=False)

print(df.columns)
X = np.array(df.drop(columns=['DeltaTemp']))
y = np.array(df['DeltaTemp'])

clf.fit(X,y)
score = clf.score(X,y)
print(clf.coef_)

print(score)

inputcolumns = ['Power','OriTemp','HeatLyeIn','HeatLyeOut','I']
SourceFolder = "20s/Original"
SourceFiles = os.listdir(SourceFolder)
SourceFiles.sort()
counter = 1
for file in SourceFiles:
    df = pandas.read_csv(os.path.join(SourceFolder,file))
    df = remove_minusV(df)
    df['Power'] = df['电解电压'] * df['电解电流']
    df['OriTemp'] = (df['氧槽温'] + df['氧槽温']) / 2
    df['HeatLyeIn'] = df['碱液流量'] * df['碱温']
    df['HeatLyeOut']    = df['碱液流量'] * df['OriTemp']
    df['I'] = df['电解电流']
    df['DeltaTemp'] = df['OriTemp'].shift(-1) - df['OriTemp']
    X = np.zeros((len(df), len(inputcolumns)))
    for col in range(len(inputcolumns)):
        X[:, col] = np.array(df[inputcolumns[col]])
    plt.subplot(6,2,counter)
    compare_DeltaandTemp(df,clf.predict(X),file,plt)

    counter = counter+1

plt.subplots_adjust(left=0.04, bottom=0.05, right=0.957, top=0.967, wspace=0.15, hspace=0.462)
plt.show()