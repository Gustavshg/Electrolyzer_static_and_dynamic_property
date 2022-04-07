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
    newdata = originaldata.drop(originaldata[originaldata['电解电压']<1E-5].index)
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



'''

'''
t0 = time.time()
SourceFolder = "20s/Original"
SelectedFiles = ["20s/Original/TJ-20211130.csv"]
SelectedFiles = os.listdir(SourceFolder)
SelectedFiles.sort()

OriginalColumns = [ '时间', '电解电压', '电解电流', '产氢量', '产氢累计量', '碱液流量', '碱温',
       '系统压力  ', '氧槽温', '氢槽温', '氧侧液位', '氢侧液位', '氧中氢', '氢中氧', '脱氧上温', '脱氧下温',
       'B塔上温', 'B塔下温', 'C塔上温', 'C塔下温', 'A塔上温', 'A塔下温', '露点', '微氧量', '出罐压力',
       '进罐温度', '进罐压力']

inputcolumns = ['Power','RadTemp','HeatLyeIn','HeatLyeOut','I']#DataSets that are imported in to model training

clf = LinearRegression(n_jobs=-1, fit_intercept=False)

inputdata = pandas.DataFrame()
outputdata = pandas.DataFrame()
coef = pandas.DataFrame()
counter =1
Ignore = 0
FlagIgnore = 0#如果为0，则不进行ignore的分析，直接显示所有的变量；如果为1，则进行ignore，只显示那四个

for file in SelectedFiles:
    Ignore = 0
    if len(SelectedFiles) == 12:
        df = pandas.read_csv(os.path.join(SourceFolder,file))
        file = os.path.join(SourceFolder,file)
    else:
        df = pandas.read_csv(file)

    if file  == "20s/Original/TJ-20210924.csv":
        if FlagIgnore == 0:
            CalTime(ToDateTime(df), 27, 27, 37)
        Ignore = 1
    elif file == "20s/Original/TJ-20211001.csv":
        CalTime(ToDateTime(df),27,26,34)
    elif file == "20s/Original/TJ-20211007.csv":
        CalTime(ToDateTime(df),26,22,35)
    elif file == "20s/Original/TJ-20211014.csv":
        CalTime(ToDateTime(df), 21, 22,24)
    elif file == "20s/Original/TJ-20211029.csv":
        CalTime(ToDateTime(df), 18, 17, 22)
    elif file == "20s/Original/TJ-20211111.csv":
        if FlagIgnore == 0:
            CalTime(ToDateTime(df), 22, 22, 5)
        Ignore = 1
    elif file == "20s/Original/TJ-20211123.csv":
        if FlagIgnore == 0:
            CalTime(ToDateTime(df),7,6.5,13)
        Ignore = 1
    elif file == "20s/Original/TJ-20211125.csv":
        if FlagIgnore == 0:
            CalTime(ToDateTime(df), 0, 10.5, 17)
        Ignore = 1
    elif file == "20s/Original/TJ-20211126.csv":
        if FlagIgnore == 0:
            CalTime(ToDateTime(df), -7, -2, 9.5)
        Ignore = 1
    elif file == "20s/Original/TJ-20211129.csv":
        if FlagIgnore == 0:
            CalTime(ToDateTime(df), -3, -3, 23)
        Ignore = 1
    elif file == "20s/Original/TJ-20211130.csv":
        if FlagIgnore == 0:
            CalTime(ToDateTime(df), 3, 1, 12)
        Ignore = 1
    elif file == "20s/Original/TJ-20211202.csv":
        if FlagIgnore == 0:
            CalTime(ToDateTime(df), -2,-3, 23)
        Ignore = 1

    if Ignore == 0 or  FlagIgnore ==0:
        df = remove_minusV(df)
        df = intercept_temp(df)
        df['Power']  = df['电解电压'] * df['电解电流'] / 1000 / 125 #这里就把数据的取值范围压缩到0-1之间
        df['OriTemp'] = ( df['氧槽温'] + df['氢槽温'])/2
        df['OriTemp'] = WL(df['OriTemp'],0.4)
        #df['RadTemp'] = (df['OriTemp'] - df['AmbT']) / 80
        df['RadTemp'] = (df['OriTemp'] +273.15 )**4 /370**4  -(df['AmbT'] +273.15 )**4 /370**4#热辐射
        df['ReverseRad'] = (df['AmbT'] +273.15 )**4 /370**4
        df['HeatLyeIn'] =  df['碱液流量'] * df['碱温'] / 140
        df['HeatLyeOut'] = df['碱液流量'] * df['OriTemp'] / 140
        df['I'] = df['电解电流'] / 1800

        df['DeltaTemp'] = df['OriTemp'].shift(-1) - df['OriTemp']
        df.dropna(inplace=True)
        X = np.zeros((len(df),len(inputcolumns)))
        for col in range(len(inputcolumns)):
            X[:,col] = np.array(df[inputcolumns[col]])

        y = np.array(df['DeltaTemp'])
        clf.fit(X, y)

        err = ERROR(df['OriTemp'],retrieve_temp(clf.predict(X),df['OriTemp'][0]))
        if len(SelectedFiles) ==12:
            if FlagIgnore == 1:
                plt.subplot(2,2,counter)
                compare_DeltaandTemp(df, clf.predict(X), file, plt)
                print(file, "Error: ", err, "Coef, ", clf.coef_)
                counter = counter + 1
            if FlagIgnore == 0:
                plt.subplot(6, 2, counter)
                compare_DeltaandTemp(df, clf.predict(X), file, plt)
                print(file, "Error: ", err, "Coef, ", clf.coef_)
                counter = counter + 1
        else:
            print(file,"Score: ",clf.score(X,y),"Coef, ",clf.coef_)
            plt.figure(figsize=(10,5))
            compare_DeltaandTemp(df,clf.predict(X),file,plt)
            plt.show()
        #data collection of all time
        inputdata = inputdata.append(pandas.DataFrame(X))
        outputdata = outputdata.append(pandas.DataFrame(y))
        coef = coef.append([[file,clf.coef_[0],clf.coef_[1],clf.coef_[2],clf.coef_[3],clf.coef_[4]]])

ip = np.array(inputdata)
op = np.array(outputdata)
clf.fit(ip,op)
print(clf.coef_[0])
coef = coef.append([[SourceFolder,clf.coef_[0][0],clf.coef_[0][1],clf.coef_[0][2],clf.coef_[0][3],clf.coef_[0][4]]])

coef.to_csv("Coefficient-0106.csv")


t2 = time.time()
print(t2-t0)
if len(SelectedFiles) ==12:
    plt.subplots_adjust(left=0.04, bottom=0.05, right=0.957, top=0.967, wspace=0.15, hspace=0.462)
    plt.show()
