'''
这里只能考虑，不一次性拟合所有的数据，因为就只有几天的模型数据比较接近，
而是选取几天工况比较简单的情况，来拟合一下，同时考虑进去历史环境温度
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
    newdata = originaldata.drop(originaldata[originaldata['电解电压']<1].index)
    newdata.index = np.arange(len(newdata))
    return newdata

def intercept_temp(originaldata):
    newdata = originaldata.drop(originaldata[originaldata['碱温'] <30].index)
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

def ERROR(OriTemp,CalResult):
    return np.sum((OriTemp- CalResult)**2 )/len(OriTemp)

def Vtn(df):
    T_ref = 25
    F = 96485
    n = 2
    CH2O = 75   #参考点状态下的水热容(单位：J/(K*mol))
    CH2  = 29
    CO2 = 29

    DHH2O =-2.86*10**5 + CH2O * (df['OriTemp'] - T_ref)    #参考点状态下的焓变(单位：J/mol)
    DHH2 = 0  + CH2 * (df['OriTemp'] - T_ref) #参考点状态下的焓变(单位：J/mol)
    DHO2 = 0  + CO2 * (df['OriTemp'] - T_ref)  #参考点状态下的焓变(单位：J/mol)
    df['Vtn'] = (DHH2 + DHO2/2 - DHH2O)/(n*F)



t0 = time.time()
SelectedFiles = ["20s/Original/TJ-20211001.csv","20s/Original/TJ-20211014.csv","20s/Original/TJ-20211029.csv","20s/Original/TJ-20211007.csv"]
inputDateFolder = "20s/ToFigure0311"
inputDateFile =  os.listdir(inputDateFolder)
OriginalColumns = [ '时间', '电解电压', '电解电流', '产氢量', '产氢累计量', '碱液流量', '碱温',
       '系统压力  ', '氧槽温', '氢槽温', '氧侧液位', '氢侧液位', '氧中氢', '氢中氧', '脱氧上温', '脱氧下温',
       'B塔上温', 'B塔下温', 'C塔上温', 'C塔下温', 'A塔上温', 'A塔下温', '露点', '微氧量', '出罐压力',
       '进罐温度', '进罐压力']
inputcolumns = ['E','RadTemp','HeatLyeIn','HeatLyeOut']#DataSets that are imported in to model training

#这一部分是要比较模型的精确程度
i=0
plt.figure(figsize=(12,12))
for file in inputDateFile:
    df = pandas.read_csv(os.path.join(inputDateFolder,file))
    plt.subplot(2,2,i+1)
    plt.xlim([0, 1])
    plt.grid()
    ax1 = plt.gca()
    a1color = '#31778E'
    ax1.set_ylim([0, 1])
    ax1.set_ylabel(r'$Electrolzer\ outlet\ temperature \ error $', color=a1color)

    mini = min(min(df['0']),min(df['2']))
    maxi = max(max(df['0']),max(df['2']))
    print((df['0']-mini)/(maxi-mini))
    line1 = ax1.scatter((df['0']-mini)/(maxi-mini),(df['2']-mini)/(maxi-mini),color = a1color)

    ax1.tick_params(axis='y', color=a1color)
    ax1.yaxis.label.set_color(a1color)



    ax2 = ax1.twinx()
    ax2.set_ylim([0, 1])
    a2color = '#5adc6c'
    mini = min(min(df['1']),min(df['3']))
    maxi = max(max(df['1']),max(df['3']))
    ax2.set_ylabel(r'$Delta \ tempearture \ error$', color=a2color)
    line2 = ax2.scatter((df['1']-mini)/(maxi-mini),(df['3']-mini)/(maxi-mini),color = a2color)
    ax2.yaxis.label.set_color(a2color)
    ax2.plot([0,1],[0,1],'r')


    i += 1

plt.subplots_adjust(left=0.055, bottom=0.062, right=0.948, top=0.948,wspace = 0.26,hspace=0.2)
plt.show()

'''#这部分是电压电流
plt.figure(figsize=(20,12))
for i in range(4):
    df = pandas.read_csv(SelectedFiles[i])
    df = remove_minusV(df)
    plt.subplot(2,2,i+1)
    plt.grid()
    ax1 = plt.gca()
    a1color = '#31778E'
    plt.xlim([-20, len(df) * 20/60 +20])
    ax1.set_ylabel(r'$Electrolyzer \ voltage \ (V)$', color = a1color)
    ax1.set_xlabel('Time \ (min)')
    line1, = ax1.plot(np.arange(len(df))*20/60, df['电解电压'],color = a1color, label = 'Electrolyzer voltage')
    ax1.tick_params(axis='y',color= a1color)
    ax1.yaxis.label.set_color(a1color)



    ax2 = ax1.twinx()
    a2color = '#5adc6c'
    ax2.set_ylabel(r'$Current \ density \ (A/m^2)$',color = a2color)
    line2, = ax2.plot(np.arange(len(df))*20/60, df['电解电流'], color = a2color,label = 'Electrolyzer voltage')
    ax2.tick_params(axis='y',color= a2color)
    ax2.yaxis.label.set_color(a2color)
'''
'''#这部分是出入口温度
plt.figure(figsize=(20,12))
for i in range(4):
    df = pandas.read_csv(SelectedFiles[i])
    df = remove_minusV(df)
    plt.subplot(2,2,i+1)
    plt.grid()
    ax1 = plt.gca()
    a1color = '#31778E'
    plt.xlim([-20, len(df) * 20/60 +20])
    ax1.set_ylabel(r'$Outlet \ temperature \ (^\circ C)$', color = a1color)
    ax1.set_ylim([10, 90])
    ax1.set_xlabel('Time \ (min)')
    line1, = ax1.plot(np.arange(len(df))*20/60, (df['氧槽温']+df['氢槽温'])/2,color = a1color, label = 'Electrolyzer voltage')
    ax1.tick_params(axis='y',color= a1color)
    ax1.yaxis.label.set_color(a1color)
    ax2 = ax1.twinx()
    a2color = '#5adc6c'
    ax2.set_ylabel(r'$Inlet \ temperature \ (^\circ C)$',color = a2color)
    line2, = ax2.plot(np.arange(len(df))*20/60, df['碱温'], color = a2color,label = 'Electrolyzer voltage')
    ax2.set_ylim([10,90])
    ax2.tick_params(axis='y',color= a2color)
    ax2.yaxis.label.set_color(a2color)
plt.subplots_adjust(left=0.033, bottom=0.057, right=0.96, top=0.967,wspace = 0.186,hspace=0.19)
plt.show()
'''



