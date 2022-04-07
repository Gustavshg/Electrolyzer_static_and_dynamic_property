# Based on the issues discovered in the last model of <Temperature model-1208>, in the 3rd section of https://lgkndmgws8.feishu.cn/docs/doccnj6oS07hvBIjSdoTcWoMhbe#olhTDh
# this model intent to resolve the major issue of this model----predicting temperature with data acquired in the last second
import os,pandas,time
import numpy as np
import openpyxl
import xlrd
import random
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split #到底在什么地方分割训练集和验证集可能是个问题，是不是每天的数据都要分割一下？
import matplotlib.pyplot as plt
from sklearn import linear_model



'''
In this section, we define functions that can be used in our program
'''
def remove9999(originaldata):
    return  originaldata.drop(originaldata[originaldata.V<=0].index)#deleting records where voltage is minus

def temp_intercept(originaldata):
    return originaldata.drop(originaldata[originaldata.OriTemp < 30].index)  # deleting records where temperature is lower than 55degree celcius

def I_intercept(originaldata):
    return originaldata.drop(originaldata[originaldata.I < 50].index)  # deleting records where temperature is lower than 50 A

def temp_retrieve(PredDeltaTemp):
    RetrievedTemp = np.empty(len(Dataset))
    RetrievedTemp[0] = float(Dataset['OriTemp'][:1])
    for delta in np.arange(1, len(PredDeltaTemp)):
        RetrievedTemp[delta] = RetrievedTemp[delta - 1] + PredDeltaTemp[delta]

    return RetrievedTemp

'''
this section is where we mark the path, and let program read all the files
    then store those files by date, to help with future data process
'''
t0 = time.time()
orifolder = "Data-original"
files = os.listdir(orifolder)
files.sort()


OfficialColumns = ['Time', 'V', 'I', 'H2-production', 'H2-production-accumulated', 'Qlye', 'Tlye', 'PreSys  ', 'TO2',
       'TH2', 'LevelO2', 'LevelH2', 'O2inH2', 'H2inO2', '脱氧上温', '脱氧下温', 'B塔上温', 'B塔下温',
       'C塔上温', 'C塔下温', 'A塔上温', 'A塔下温', 'DwePoint', '微氧量', '出罐压力', '进罐温度', '进罐压力']


DataReadFlag = 1 #是否需要一个读取的过程,这里读取与写入的都是原始的数据，不进行处理

if DataReadFlag ==1:#把所有的单日的数据，放到一个文件里面
    for folder in files:#首先读取大文件夹中的序列，然后读取里面所有20s的数据
        subfolder = os.path.join(orifolder, folder)#this line is compulsory, and if missed, isfile function will not be able to return the desired result
        subfiles = os.listdir(subfolder)
        subfiles.sort()
        all_data_20s = pandas.DataFrame()#可能需要在这里生命dataframe，这样每个文件夹都是一个新的dataframe
        counter_20s =0
        for file20s in subfiles:
            subpath = os.path.join(subfolder, file20s)
            all_data_2s = pandas.DataFrame()#可能需要在这里生命dataframe，这样每个文件夹都是一个新的dataframe
            if os.path.isfile(subpath):
                pd = pandas.read_excel(subpath)
                all_data_20s = all_data_20s.append(pd)
            if os.path.isdir(subpath):
                files2s = os.listdir(subpath)
                files2s.sort()
                counter_2s = 0
                for file2s in files2s:
                    pd2s = pandas.read_excel(os.path.join(subpath,file2s))
                    all_data_2s = all_data_2s.append(pd2s)
                    counter_2s = counter_2s +1
                if counter_2s == len(files2s):
                    all_data_2s.index = np.arange(all_data_2s.shape[0])
                    #all_data_2s.columns = OfficialColumns
                    all_data_2s.to_csv(os.path.join("./2s/Original", folder + ".csv"))
            counter_20s = counter_20s+1
            if counter_20s == len(subfiles):
                all_data_20s.index = np.arange(all_data_20s.shape[0])
                #all_data_20s.columns = OfficialColumns
                all_data_20s.to_csv(os.path.join("./20s/Original",folder+".csv"))

'''
这里读取之前写入的所有数据，然后去除不合适的部分，把新的文件写入Modified文件夹
'''
OriFolders = ['20s','2s']#这里有两个文件夹里面的东西都要读写

ModifierFlag = 0 #如果为1，就读取原始数据，处理后在输出到Modified文件夹

if ModifierFlag ==1:
    for folder in OriFolders:
        orifilepath = os.path.join(folder,"Original")
        orifiles = os.listdir(orifilepath)
        orifiles.sort()
        ModiDataFrame = pandas.DataFrame()
        counter = 0
        for file in orifiles:#每读取一个原始的文件，就需要输出一个modified文件
            pd = pandas.read_csv(os.path.join(orifilepath,file))
            ModiDataFrame = remove9999(pd)
            ''' 这里加入一些新的变量，并且不做平滑处理'''
            ModiDataFrame['OriTemp'] = (ModiDataFrame['TO2']+ModiDataFrame['TH2'])/2
            ModiDataFrame['Power'] = ModiDataFrame['V']*ModiDataFrame['I']
            ModiDataFrame['HeatLyeIn'] = ModiDataFrame['Qlye'] * ModiDataFrame['Tlye']
            ModiDataFrame['HeatLyeOut'] = ModiDataFrame['Qlye']*ModiDataFrame['OriTemp']


            ModiDataFrame.to_csv(os.path.join("./"+folder+"/Modified",file))
            counter = counter+1

t1 = time.time()
print('the time consumed in data IO is', t1-t0)

'''
这部分需要读取20s或者2s文件夹的数据，并且进行处理，因为要所有的文件训练在同一个模型里，因此可能需要不停循环进行训练
'''
DataSelection = ['20s']# 20s or 2s
inputcolumns = ['Power','OriTemp','HeatLyeIn','HeatLyeOut','I']#DataSets that are imported in to model training
outputcolumns = ['DeltaTemp'] #columns required in the output region
predictlength = 1#预测长度为1步，具体的时间取决于使用的数据形式


TrainFlag = 0 #如果为1，则开始模型的训练
RandomFlag = 1 #如果为1，则进行随机化处理
'''
如果为1，则进行数据预处理；这里的预处理可能有问题，因为每个日期的数据，预处理的标准可能不一样，所以模型训练的过程也就不一样，因此不能在不同的数据组之间进行预处理，最后还训练一个模型
'''
PreprocessFlag = 0
TrainTestSplitFlag = 1#如果为1，则区分训练集和验证集

if TrainFlag ==1:
    OriFiles = os.listdir(os.path.join("./"+DataSelection[0],"Modified"))
    OriFiles.sort()
    clf = LinearRegression(n_jobs=-1)
    #clf = linear_model.Lasso(alpha = 0.1)
    for file in OriFiles[:1]:#每一个文件的单独训练都在这个for循环里面
        Dataset = pandas.read_csv(os.path.join("./"+DataSelection[0],"Modified/"+file))
        Dataset['PredTemp'] = Dataset['OriTemp'].shift(-predictlength)
        Dataset['DeltaTemp'] = Dataset['PredTemp'] - Dataset['OriTemp']
        Dataset = temp_intercept(Dataset)
        Dataset = I_intercept(Dataset)
        Dataset.dropna(inplace=True)
        X = np.array(Dataset[inputcolumns[0]]).reshape(len(Dataset),1)
        for col in inputcolumns[1:]:
            X = np.concatenate((X, np.array(Dataset[col]).reshape(len(Dataset), 1)), axis=1)
        for col in outputcolumns:
            y = np.array(Dataset[col]).reshape(len(Dataset), 1)

        if PreprocessFlag ==1:
            X = preprocessing.scale(X)

        X_input = X.copy()
        y_input = y.copy()

        if RandomFlag == 1:
            random.seed(124134)
            random.shuffle(X_input)
            random.shuffle(y_input)

        if TrainTestSplitFlag ==1:
            X_train, X_test, y_train, y_test = train_test_split(X_input, y_input, test_size=0.25)#在每一天的数据里面，还是区分一下训练集和验证集
        else:
            X_train = X_input
            y_train = y_input
        '''
Training Section
        '''

        clf.fit(X_train,y_train)

        confidence = clf.score(X, y)
        SampleFlag = 0
        if SampleFlag ==1:
            ToCSVData = pandas.DataFrame(y)
            ToCSVData.to_csv('output.csv')
            ToCSVData = pandas.DataFrame(X)
            ToCSVData.to_csv('input.csv')

        if TrainTestSplitFlag ==1:
            confidence = clf.score(X_test,y_test)
        else:
            confidence = clf.score(X_train,y_train)


    print("The confidence of total model is", confidence)

t2 = time.time()
print('The time consumed in model training is', t2-t1)

'''
这部分想画一下每天温度的变化过程，和预测的值的对比
'''






RetrieveFlag = 0#如果为1，则进行温度复现过程

if RetrieveFlag == 1 & TrainFlag ==1:
    OriFiles = os.listdir(os.path.join("./" + DataSelection[0], "Modified"))
    OriFiles.sort()
    for file in OriFiles[5:6]:
        Dataset = pandas.read_csv(os.path.join("./" + DataSelection[0], "Modified/" + file))
        Dataset['PredTemp'] = Dataset['OriTemp'].shift(-predictlength)
        Dataset['DeltaTemp'] = Dataset['PredTemp'] - Dataset['OriTemp']
        #plt.plot(Dataset['I'], label=file+'before intercept')
        Dataset = temp_intercept(Dataset)
        Dataset = I_intercept(Dataset)
        Dataset.dropna(inplace=True)
        Dataset.index = np.arange(Dataset.shape[0])

        X = np.array(Dataset[inputcolumns[0]]).reshape(len(Dataset), 1)
        for col in inputcolumns[1:]:
            X = np.concatenate((X, np.array(Dataset[col]).reshape(len(Dataset), 1)), axis=1)

        if PreprocessFlag ==1:
            X = preprocessing.scale(X)

        y=[]
        for x in X:
            y.append(np.vdot(x,RightCoef)+RightIntercept)




        PredDeltaTemp = clf.predict(X)


        PltFlag = 1#如果是1，就输出温度差结果，如果不是1，就输出累积温度的结果
        if PltFlag ==1:
            plt.plot(Dataset['DeltaTemp'], label=file)
            plt.plot(PredDeltaTemp,label = "Predicted "+file)

        elif PltFlag ==0:
            plt.plot(Dataset['OriTemp'], label=file)
            RetrievedTemp = temp_retrieve(PredDeltaTemp)
            plt.plot(RetrievedTemp,label = "Predicted "+file)
        elif PltFlag == 2:
            plt.plot(Dataset['OriTemp'], label=file)
            plt.title("Original Temperature")


    t3 = time.time()
    print('The coeffficients of the model is ',clf.intercept_)
    print('The time consumed in predicting is ', t3-t2)


    plt.legend()
    plt.show()
