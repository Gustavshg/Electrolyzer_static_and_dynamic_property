#模型前面的框架还是可以沿用之前的1214的版本，重新进行一下文件的读写，看看有没有什么大问题
'''
这里有一个命名规则要注意下，所有的函数，因该是动词在前，宾语在后；所有的变量，应该是名词在前，形容词在后
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
    newdata = originaldata.drop(originaldata[originaldata.V<0.5].index)
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

'''
第一部分是读取原始的数据，将每一天的数据合并之后，写入一个单独的文件中，文件夹为“Original”
'''
t0 = time.time()# initial time
SourceFolder = "Data-original"# folder from which all the files are loaded
OfficialColumns = ['Time', 'V', 'I', 'H2-production', 'H2-production-accumulated', 'Qlye', 'Tlye', 'PreSys  ', 'TO2',
       'TH2', 'LevelO2', 'LevelH2', 'O2inH2', 'H2inO2', '脱氧上温', '脱氧下温', 'B塔上温', 'B塔下温',
       'C塔上温', 'C塔下温', 'A塔上温', 'A塔下温', 'DwePoint', '微氧量', '出罐压力', '进罐温度', '进罐压力']
#Official columns used in data
FlagDataRead = 1 #是否需要一个读取的过程,这里读取与写入的都是原始的数据，不进行处理

if FlagDataRead == 1:#把所有的单日的数据，放到一个文件里面
    folders = os.listdir(SourceFolder)#这里的每一个自路径都是一个文件夹
    folders.sort()
    for folder in folders:#首先读取大文件夹中的序列，然后读取里面所有20s的数据
        subfolder = os.path.join(SourceFolder,folder)
        subfiles = os.listdir(subfolder)
        subfiles.sort()
        Data_20s = pandas.DataFrame()#可能需要在这里生命dataframe，这样每个文件夹都是一个新的dataframe
        counter_20s = 0#这个计数器用来计算是否已经遍历完了每个日期下面的所有一级子文件（2s文件夹和20s数据文件）
        for file_20s in subfiles:
            subpath = os.path.join(subfolder,file_20s)
            Data_2s = pandas.DataFrame()#可能需要在这里生命dataframe，这样每个文件夹都是一个新的dataframe
            counter_20s = counter_20s+1
            if os.path.isfile(subpath):#如果这里是文件，那就是需要读取的20s的数据文件
                pd_20s = pandas.read_excel(subpath)
                Data_20s = Data_20s.append(pd_20s)
            elif os.path.isdir(subpath):#如果这里是路径，那就说明是2s的子文件夹，要进去读取每个子文件夹里面的子文件
                files_2s = os.listdir(subpath)
                files_2s.sort()
                counter_2s = 0#对于2s的文件来说，for循环结束之后，自然就是一个日期的结束了，就没有必要在用counter了
                for file_2s in files_2s:#这里读取所有的2s的数据文件，放到一个文件里面
                    pd_2s = pandas.read_excel(os.path.join(subpath,file_2s))
                    Data_2s = Data_2s.append(pd_2s)
                Data_2s.index = np.arange(len(Data_2s))
                #Data_2s.columns = OfficialColumns
                Data_2s.to_csv(os.path.join("./2s/Original",folder+".csv"))
        if counter_20s == len(subfolder):
            Data_20s.index = np.arange(len(Data_20s))
            #Data_20s.columns = OfficialColumns
            Data_20s.to_csv(os.path.join("./20s/Original",folder + ".csv"))
    t1 = time.time()
    print('The time consumed in data IO is',t1-t0)

'''
这里读取之前写入的所有数据，然后去除不合适的部分，把新的文件写入Modified文件夹
'''
t1 = time.time()
SourceFolder = ['20s','2s']#这里有两个文件夹里面的东西都要读写
predictlength = 1#预测长度为1步，具体的时间取决于使用的数据形式
inputcolumns = ['Power','OriTemp','HeatLyeIn','HeatLyeOut','I']#DataSets that are imported in to model training
Threshold = 0.5

FlagModify = 0 #如果为1，就读取原始数据，处理后在输出到Modified文件夹

if FlagModify ==1:
    for folder in SourceFolder:
        orifilepath = os.path.join(folder,"Original")
        orifiles = os.listdir(orifilepath)
        orifiles.sort()
        ModiDataFrame = pandas.DataFrame()
        counter = 0

        for file in orifiles:#每读取一个原始的文件，就需要输出一个modified文件
            pd_modi = pandas.read_csv(os.path.join(orifilepath,file))
            ModiDataFrame = remove_minusV(pd_modi)
            ''' 这里加入一些新的变量，并且不做平滑处理'''
            ModiDataFrame['OriTemp'] = (ModiDataFrame['TO2'] + ModiDataFrame['TH2']) / 2
            ModiDataFrame['Power'] = ModiDataFrame['V'] * ModiDataFrame['I']
            ModiDataFrame['HeatLyeIn'] = ModiDataFrame['Qlye'] * ModiDataFrame['Tlye']
            ModiDataFrame['HeatLyeOut'] = ModiDataFrame['Qlye'] * ModiDataFrame['OriTemp']
            for col in inputcolumns:#对于所有需要被输入到模型的参数，都进行平滑处理
                ModiDataFrame[col] = WL(ModiDataFrame[col],Threshold)
            ModiDataFrame.to_csv(os.path.join("./" + folder + "/Modified",file))
            counter =counter+1


    t2 = time.time()
    print('The time consumed in data modifying is', t2-t1)

'''
这部分需要读取20s或者2s文件夹的数据，并且进行处理，因为要所有的文件训练在同一个模型里，因此可能需要不停循环进行训练
'''
t2 = time.time()
DataSelection = ['20s']# 20s or 2s
#inputcolumns = ['Power','OriTemp','HeatLyeIn','HeatLyeOut','I']#DataSets that are imported in to model training
outputcolumns = ['DeltaTemp'] #columns required in the output region
predictlength = 1#预测长度为1步，具体的时间取决于使用的数据形式

FlagTrain = 0 #如果为1，则开始模型的训练
FlagRandom = 0 #如果为1，则进行随机化处理
FlagSplit = 0 #如果为1，则区分训练集和验证集
FlagSampling = 0 #如果为1，就进行数据采样输出

coef = pandas.DataFrame(columns=['file','1','2','3','4','5'])
print(coef)




#clf = linear_model.Lasso(alpha = 0.5)
if FlagTrain ==1:
    '''今天先简单一点，只读取一个文件，然后进行模型训练，看能不能得到正确的结果'''
    SamplingData = pandas.DataFrame()
    ModiFiles = os.listdir(os.path.join("./" + DataSelection[0],"Modified"))
    ModiFiles.sort()
    counter = 1
    for file in ModiFiles[:]:#每一个文件的单独训练都在这个for循环里面
        clf = LinearRegression(fit_intercept=False, n_jobs=-1)
        DataSet = pandas.read_csv(os.path.join("./"+DataSelection[0],"Modified/"+file))
        DataSet['PredTemp'] = DataSet['OriTemp'].shift(-predictlength)
        DataSet['DeltaTemp'] = DataSet['PredTemp'] - DataSet['OriTemp']
        DataSet.dropna(inplace=True)
        '''以上能够用到的数据，应该都已经经过了平滑,小波分解对于阶跃的数据，没法很好做变换，这个可能要注意'''
        X = np.zeros((len(DataSet), len(inputcolumns)))
        y = np.zeros((len(DataSet), 1))
        for col in range(len(inputcolumns)):
            X[:, col] = np.array(DataSet[inputcolumns[col]])
        for col in range(len(outputcolumns)):
            y = np.array(DataSet[outputcolumns[col]]).reshape(len(DataSet), 1)

        data = pandas.DataFrame(X)
        data.columns = inputcolumns
        data['DeltaTemp'] = y
        SamplingData=SamplingData.append(data,ignore_index=True)


        X_input = X.copy()
        y_input = y.copy()

        if FlagSplit == 1:
            X_train, X_test, y_train, y_test = train_test_split(X_input, y_input, test_size=0.25)
        else:
            X_train = X_input
            y_train = y_input
            X_test = X_input
            y_test = y_input

        clf.fit(X_train, y_train)

        #coef = coef.append([file,clf.coef_[0][0],clf.coef_[0][1],clf.coef_[0][2],clf.coef_[0][3],clf.coef_[0][4]])
        coef.loc[counter-1] = [file,clf.coef_[0][0],clf.coef_[0][1],clf.coef_[0][2],clf.coef_[0][3],clf.coef_[0][4]]
        #print("the coef_ of model with <" + file + "> is " + str(clf.coef_[0]))
        plt.subplot(6, 2, counter)
        compare_DeltaandTemp(DataSet,clf.predict(X),file,plt)
        counter = counter+1

    if FlagSampling == 1:
        SamplingData.to_csv("Threshold = "+str(Threshold)+" -alldata.csv")

        FlagCompare =0
        if FlagCompare ==1:
            DataSet = SamplingData
            X = DataSet[inputcolumns]
            y = DataSet[outputcolumns]
            clf.fit(X,y)
            PredDeltaTemp = clf.predict(X)
            compare_DeltaandTemp(DataSet, PredDeltaTemp, file, plt)
    '''训练完了模型，现在对每一个文件的温度进行复盘'''
    mse = 0
    # for file in ModiFiles[:]:  # 每一个文件的单独训练都在这个for循环里面
    #     DataSet = pandas.read_csv(os.path.join("./" + DataSelection[0], "Modified/" + file))
    #     DataSet['PredTemp'] = DataSet['OriTemp'].shift(-predictlength)
    #     DataSet['DeltaTemp'] = DataSet['PredTemp'] - DataSet['OriTemp']
    #     DataSet.dropna(inplace=True)
    #
    #     X = np.zeros((len(DataSet), len(inputcolumns)))
    #     y = np.zeros((len(DataSet), 1))
    #
    #     for col in range(len(inputcolumns)):
    #         X[:, col] = np.array(DataSet[inputcolumns[col]])
    #     for col in range(len(outputcolumns)):
    #         y = np.array(DataSet[outputcolumns[col]]).reshape(len(DataSet), 1)
    #
    #
    #     PredDeltaTemp = clf.predict(X)
    #
    #     Error = np.array(DataSet['OriTemp'] - retrieve_temp(PredDeltaTemp,DataSet['OriTemp'][0]))
    #     for i in range(len(Error)):
    #         mse = mse + Error[i]**2
        #compare_DeltaandTemp(DataSet,PredDeltaTemp,file,plt)


    print("the MSE of the model is " + str(mse))
    t3 = time.time()
    print(coef)
    coef.to_csv("Threshold=" +str(Threshold)+" coefs.csv")
    print("the time consumed in model training is "+ str(t3-t2))

    plt.subplots_adjust(left=0.04, bottom=0.05, right=0.957, top=0.967, wspace=0.15, hspace=0.462)
    plt.show()
'''
这部分想画一下温度的变化过程，和预测的值的对比
'''
FlagRetrieve =0

if FlagTrain ==1 & FlagRetrieve ==1:
    PredDeltaTemp = clf.predict(X)

    compare_DeltaandTemp(DataSet,PredDeltaTemp,plt)