#模型前面的框架还是可以沿用之前的1216的版本
#主要是读取文件过程变成读取smooth文件夹内容，同时在恢复温度时，逐天进行而不是整体恢复，函数等还是可以沿用之前的版本
'''
这里有一个命名规则要注意下，所有的函数，因该是动词在前，宾语在后；所有的变量，应该是名词在前，形容词在后
'''
import os, pandas , time, xlrd, openpyxl,random
import numpy as np
import matplotlib.pyplot as plt
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
因为这里不需要我们读取所有之前的文件了，所以直接读取smooth文件就可以
如果以后有需要的话，是在不行可以用1216的文件来进行文件的读写
'''

t0 = time.time()

inputcolumns = ['Power','OriTemp','HeatLyeIn','HeatLyeOut','I']#DataSets that are imported in to model training
outputcolumns = ['DeltaTemp'] #columns required in the output region
predictlength = 1#预测长度为1步，具体的时间取决于使用的数据形式

SourceFolder = "20s/Smooth"
SourceFiles = os.listdir(SourceFolder)
SourceFiles.sort()
#clf = LinearRegression(fit_intercept=False,n_jobs=-1)
#clf = linear_model.ElasticNet(fit_intercept=False)
#clf = linear_model.Ridge(alpha=0.1,fit_intercept=False)
#clf = linear_model.Lars(fit_intercept=False)
#clf = linear_model.OrthogonalMatchingPursuit(n_nonzero_coefs=5,fit_intercept=False)
#clf = linear_model.LogisticRegression(fit_intercept=False)
#clf = linear_model.SGDRegressor(fit_intercept=False,epsilon=0.1)
clf = linear_model.Perceptron(fit_intercept=False,n_jobs=-1)

#RightCoef = [2.67136251633959e-06, -0.000669124338003471, 0.00783019047295691, -0.00919934147979449, 9.25647676465811e-05]


for file in SourceFiles[6:7 ]:
    SourceFile = os.path.join(SourceFolder,file)
    SourceData = pandas.read_csv(SourceFile)

    X = np.array(SourceData[inputcolumns])
    y = np.array(SourceData[outputcolumns]).reshape(len(SourceData),)


    clf.fit(X,y)
    #clf.coef_[0] = RightCoef
    print(clf.coef_)

    score = clf.score(X,y)
    print("the score of the model in <"+SourceFile +"> is "+str(score))

    RecoverFolder = "20s/Modified"
    RecoverFiles = os.listdir(RecoverFolder)
    RecoverFiles.sort()
    num = len(RecoverFiles)
    counter =1

    for RecoverFile in RecoverFiles:
        DataSet = pandas.read_csv(os.path.join(RecoverFolder,RecoverFile))
        DataSet['PredTemp'] = DataSet['OriTemp'].shift(-predictlength)
        DataSet['DeltaTemp'] = DataSet['PredTemp'] - DataSet['OriTemp']
        DataSet.dropna(inplace=True)



        X = np.array(DataSet[inputcolumns])
        plt.subplot(6,2,counter)
        compare_DeltaandTemp(DataSet,clf.predict(X),RecoverFile,plt)
        counter=counter+1
    plt.subplots_adjust(left=0.04,bottom=0.05,right=0.957,top=0.967,wspace=0.15,hspace=0.462)
    plt.show()




t1 = time.time()
print("the time consumed in model training is " + str(t1-t0))



