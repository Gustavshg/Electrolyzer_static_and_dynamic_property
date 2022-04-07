'''
之前的模型里面其实都有一个问题，就是碱液进出口训练出来的系数，其实都不太一样
虽然说差的都不太多，但是还是有一点差异，这次尝试一下将HeatLyeIn & HeatLyeOut合并成一项在进行尝试
'''

import os, pandas , time, xlrd, openpyxl,random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pywt #python wavelet transmission
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import linear_model




'''
这部分定义一下各种函数，应该尽可能对于重复的部分多用一些函数
'''
def remove_minusV(originaldata):
    #删除电压小于零的部分，有一些可能是无用数据
    newdata = originaldata.drop(originaldata[originaldata['电解电压']<0.5].index)
    newdata.index = np.arange(len(newdata))
    return newdata

def intercept_temp(originaldata):
    newdata = originaldata.drop(originaldata[originaldata.TO2<30].index)
    newdata.index = np.arange(len(newdata))
    return newdata

def intercept_I(originaldata):
    newdata = originaldata.drop(originaldata[originaldata.I<0].index)
    newdata.index = np.arange(len(newdata))
    return newdata

def retrieve_temp(DeltaTempPredicted,InitialTemp):
    TempRetrieved = np.empty(len(DeltaTempPredicted))
    TempRetrieved[0] = float(InitialTemp)
    for i in np.arange(1,len(TempRetrieved)):
        TempRetrieved[i] = TempRetrieved[i-1] + DeltaTempPredicted[i]
    return TempRetrieved

def EMA(data,beta=0.5):#进行指数滑动平均，进行滤波
    theta = np.array(data).reshape(len(data),1)
    v = np.zeros((len(data),1))
    v[0] = theta[0]
    for t in np.arange(1,len(data)):
        v[t] = v[t-1]*beta + (1-beta)*theta[t]
    return v

def AA(data,step = 1):#算术平均滤波法
    v = np.zeros((len(data)))
    for t in np.arange(step+1):
        v[t] = np.average(data[:t+step])
        v[-t] = np.average(data[-(t+step):])
    for t in np.arange(step+1
            ,len(data)-step+1):
        v[t] = np.average(data[(t-step):(t+step)])

    return v

def WL(data_input,threshold = 0.3):#小波分解
    index = []
    data = []
    for i in range(len(data_input) - 1):
        X = float(i)
        Y = float(data_input[i])
        index.append(X)
        data.append(Y)
    # 创建小波对象并定义参数:
    w = pywt.Wavelet('db8')  # 选用Daubechies8小波
    maxlev = pywt.dwt_max_level(len(data), w.dec_len)
    coeffs = pywt.wavedec(data, 'db8', level=maxlev)
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold * max(coeffs[i]))
    data_output = pywt.waverec(coeffs, 'db8')
    if len(data_output)!= len(data_input):
        data_output = np.append(data_output, data_output[len(data_output) - 1])
    return data_output

def MSE(ps_filter_sample):

    mse = 0
    for i in range(len(ps_filter_sample['OriTemp'])):
        mse = mse + np.sqrt((ps_filter_sample['OriTemp'][i]-ps_filter_sample['SmoothTemp'][i])**2)
    mse = mse/len(ps_filter_sample)
    return mse

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

def original_data_exam(DataSet,file,plt):
    ax1 = plt.gca()
    plt.xlim([-20,len(DataSet)*1.1])
    print(file)
    ax1.set_ylabel(r'$Temperature \quad \circ_C$')
    ax1.set_xlabel('Time')
    ax1.plot(np.arange(len(DataSet)),DataSet['氧槽温'],color = 'brown')
    ax1.plot(np.arange(len(DataSet)),DataSet['氢槽温'],color ='orange')
    ax1.plot(np.arange(len(DataSet)),DataSet['碱温'],color ='greenyellow')
    ax1.legend(['T O2','T H2','T Lye'],loc=1)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Current(1000A) & Lye Flux')
    ax2.plot(np.arange(len(DataSet)),DataSet['电解电流']/1000,color ='turquoise')
    ax2.plot(np.arange(len(DataSet)),DataSet['碱液流量'],color ='darkviolet')
    ax2.legend(['Current','Lye Fluex'],loc =8)
    plt.title(file)



'''
这次尝试不再进行文件的原始处理，所以直接读取original文件就可以了
'''
t0 = time.time()
SourceFolder = "2s/Original"
SourceFiles = os.listdir(SourceFolder)
SourceFiles.sort()

OriginalColumns = ['时间', '电解电压', '电解电流', '产氢量', '产氢累计量', '碱液流量', '碱温',
       '系统压力  ', '氧槽温', '氢槽温', '氧侧液位', '氢侧液位', '氧中氢', '氢中氧', '脱氧上温', '脱氧下温',
       'B塔上温', 'B塔下温', 'C塔上温', 'C塔下温', 'A塔上温', 'A塔下温', '露点', '微氧量', '出罐压力',
       '进罐温度', '进罐压力']
inputcolumns = ['Power','OriTemp','DeltaHeatLye', 'I','电解电压']


Threshold = 0.1

counter =1
coef = pandas.DataFrame()
for file in SourceFiles:
    clf = LinearRegression(fit_intercept=False)
    df = pandas.read_csv(os.path.join(SourceFolder,file))
    df = remove_minusV(df)



    df['Power'] = df['电解电压'] * df['电解电流']
    df['OriTemp'] = ((df['氧槽温'] + df['氧槽温']) / 2 - 20)#这里代入的是与环境温度的差值
    df['DeltaHeatLye'] = df['碱液流量']* ( df['碱温'] - df['OriTemp'])
    df['I'] = df['电解电流']

    for col in inputcolumns:
        df[col] = WL(df[col],Threshold)

    df['DeltaTemp'] = df['OriTemp'].shift(-1) - df['OriTemp']
    df.dropna(inplace=True)



    X = np.zeros((len(df),len(inputcolumns)))
    y = np.array(df['DeltaTemp'])
    for col in range(len(inputcolumns)):
        X[:,col] = np.array(df[inputcolumns[col]])

    clf.coef_ = [1.06451552e-05, -4.47839759e-03, 145177748e-03, -4.59249787e-04]
    clf.fit(X,y)
    print(clf.coef_)

    coef =coef.append( [[file,clf.coef_[0],clf.coef_[1],clf.coef_[2],clf.coef_[3]]])
    #PredDeltaTemp = clf.predict(X)
    PredDeltaTemp = clf.coef_[0] * df['Power']  +clf.coef_[1]  * df['OriTemp'] + clf.coef_[2] * df['DeltaHeatLye'] + clf.coef_[3] * df['I']
    plt.subplot(6, 2, counter)
    compare_DeltaandTemp(df,PredDeltaTemp,file,plt)



    counter = counter + 1
coef.to_csv('coefficient-1228.csv')

plt.subplots_adjust(left=0.04, bottom=0.05, right=0.957, top=0.967, wspace=0.15, hspace=0.462)
plt.show()

















