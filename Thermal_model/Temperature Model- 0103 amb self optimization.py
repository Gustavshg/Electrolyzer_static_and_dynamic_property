'''
这里只能考虑，不一次性拟合所有的数据，因为就只有几天的模型数据比较接近，
而是选取几天工况比较简单的情况，来拟合一下，同时考虑进去历史环境温度
'''
'''
这里的内容总体沿袭自1230文件，但是这里想尝试一下，在一定范围内自优化环境温度，以得到最适宜的环境温度选项
'''
import datetime
import os, pandas , time, xlrd, openpyxl,random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pywt #python wavelet transmission
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import math

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

def ToDateTime(df):#可以把中文日期，转化为可读的日期格式
    new_df = pandas.DataFrame()
    for date in df['时间']:
        tt = pandas.to_datetime(date.replace('年','-').replace('月','-').replace('日',' ').replace('时',':').replace('分',':').replace('秒',''))
        tt = tt.strftime('%H%M%S')
        new_df = new_df.append([tt])
    new_df.index = range(len(df))
    df['Time'] = new_df
    return df


def CalTime(df,BoT,EoT,TMAX):#这里计算每一个时间到当时凌晨两点的秒数，然后可以进行下一步环境温度的计算
    AMBT = pandas.DataFrame()
    for tick in df['Time']:
        dt = int(tick[:2]) * 3600 + int(tick[2:4]) * 60 + int(tick[4:]) - 2 * 3600# seconds from this day 02:00:00
        ambT = BoT + (EoT-BoT)*dt/86400 + (TMAX - (EoT+BoT)/2)/2 *( math.sin(dt/86400*2*math.pi - math.pi/2)+1)
        AMBT = AMBT.append(pandas.DataFrame(np.array([ambT])))
    AMBT.index = range(len(AMBT))
    df['AmbT'] = AMBT

def TempOptimize(BoT,EoT,TMAX,range=10.,step = 4.):
    BoTArray = np.arange(float(BoT-range),float(BoT+range),range/step)
    EoTArray = np.arange(float(EoT - range), float(EoT + range), range / step)
    TMAXArray = np.arange(float(TMAX - range), float(TMAX + range), range / step)


    return BoTArray,EoTArray,TMAXArray

'''
FUNCTION TEST
'''
t0 = time.time()

if 1:
    OriginalColumns = ['时间', '电解电压', '电解电流', '产氢量', '产氢累计量', '碱液流量', '碱温',
                       '系统压力  ', '氧槽温', '氢槽温', '氧侧液位', '氢侧液位', '氧中氢', '氢中氧', '脱氧上温', '脱氧下温',
                       'B塔上温', 'B塔下温', 'C塔上温', 'C塔下温', 'A塔上温', 'A塔下温', '露点', '微氧量', '出罐压力',
                       '进罐温度', '进罐压力']

    inputcolumns = ['Power', 'RadTemp', 'HeatLyeIn', 'HeatLyeOut',
                    'I']  # DataSets that are imported in to model training
    clf = LinearRegression(n_jobs=-1, fit_intercept=False)
    Date = ["0924","1001","1007","1014","1029","1111","1123","1125","1126","1129","1130","1202"]
    for date in Date:
        SelectedFiles = "20s/Original/TJ-2021" + date + ".csv"
        df = pandas.read_csv(SelectedFiles)
        df = remove_minusV(df)
        ResultPD = pandas.DataFrame()
        if SelectedFiles == "20s/Original/TJ-20210924.csv":
            BoTArray, EoTArray, TMAXArray = TempOptimize( 20, 18, 25)
        elif SelectedFiles == "20s/Original/TJ-20211001.csv":
            BoTArray, EoTArray, TMAXArray = TempOptimize( 19, 20, 28)
        elif SelectedFiles == "20s/Original/TJ-20211007.csv":
            BoTArray, EoTArray, TMAXArray = TempOptimize( 11, 12, 17)
        elif SelectedFiles == "20s/Original/TJ-20211014.csv":
            BoTArray, EoTArray, TMAXArray = TempOptimize( 12, 15, 20)
        elif SelectedFiles == "20s/Original/TJ-20211029.csv":
            BoTArray, EoTArray, TMAXArray = TempOptimize( 10, 8, 23)
        elif SelectedFiles == "20s/Original/TJ-20211111.csv":
            BoTArray, EoTArray, TMAXArray = TempOptimize( 4, 3, 18)
        elif SelectedFiles == "20s/Original/TJ-20211123.csv":
            BoTArray, EoTArray, TMAXArray = TempOptimize( 2, -1, 20)
        elif SelectedFiles == "20s/Original/TJ-20211125.csv":
            BoTArray, EoTArray, TMAXArray = TempOptimize( 0, 3, 17)
        elif SelectedFiles == "20s/Original/TJ-20211126.csv":
            BoTArray, EoTArray, TMAXArray = TempOptimize( 3, 2, 17)
        elif SelectedFiles == "20s/Original/TJ-20211129.csv":
            BoTArray, EoTArray, TMAXArray = TempOptimize( -3, -3, 23)
        elif SelectedFiles == "20s/Original/TJ-20211130.csv":
            BoTArray, EoTArray, TMAXArray = TempOptimize(3, 1, 12)
        elif SelectedFiles == "20s/Original/TJ-20211202.csv":
            BoTArray, EoTArray, TMAXArray = TempOptimize( -2, -3, 23)
        #BoTArray, EoTArray, TMAXArray = TempOptimize(20,18,25,10,3)
        CalSize = len(BoTArray)*len(EoTArray)*len(TMAXArray)
        counter = 1
        for bot in BoTArray:
            for eot in EoTArray:
                for tmax in TMAXArray:
                    CalTime(ToDateTime(df), bot, eot, tmax)
                    df['Power'] = df['电解电压'] * df['电解电流'] / 1000 / 125  # 这里就把数据的取值范围压缩到0-1之间
                    df['OriTemp'] = (df['氧槽温'] + df['氢槽温']) / 2
                    df['OriTemp'] = WL(df['OriTemp'], 0.2)
                    df['RadTemp'] = (df['OriTemp'] - df['AmbT']) / 80  # 热辐射温度就是工作的槽温减去环境温度
                    df['HeatLyeIn'] = df['碱液流量'] * df['碱温'] / 140
                    df['HeatLyeOut'] = df['碱液流量'] * df['OriTemp'] / 140
                    df['I'] = df['电解电流'] / 1800

                    df['DeltaTemp'] = df['OriTemp'].shift(-1) - df['OriTemp']
                    df.dropna(inplace=True)
                    X = np.zeros((len(df), len(inputcolumns)))
                    for col in range(len(inputcolumns)):
                        X[:, col] = np.array(df[inputcolumns[col]])

                    y = np.array(df['DeltaTemp'])
                    clf.fit(X, y)
                    score = clf.score(X,y)
                    ResultPD = ResultPD.append(([[bot,eot,tmax,score,clf.coef_[0],clf.coef_[1],clf.coef_[2],clf.coef_[3],clf.coef_[4]]]))


                    print("Score:",score," progress:",counter,"/",CalSize,", time comsumed:", time.time()-t0)

                    counter = counter +1
        ResultPD.columns = ["BoT","EoT","TMAX","Score","Power","OriTemp","HeatLyeIn","HeatLyeOut","I"]
        path = "TempSelfOpt of "+ date+".csv"
        ResultPD.to_csv(path)
