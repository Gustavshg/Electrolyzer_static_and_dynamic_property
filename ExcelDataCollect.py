
import os
import pandas
import numpy as np
import xlrd
import matplotlib
import openpyxl

#这个程序是针对2021年11月底在新乡开展的电解槽测试的数据处理及最后模型搭建的问题
#最终的目的是想建立包含电解槽各种动态特性响应的数据驱动模型
former = 0
if former == 1:
    orifolder = "Data-original"
    files = os.listdir(orifolder)
    files.sort()
    all_data_20s = pandas.DataFrame()
    all_data_2s = pandas.DataFrame()
    OfficialColumns = ['Time', 'V', 'I', 'H2-production', 'H2-production-accumulated', 'Qlye', 'Tlye', 'PreSys  ', 'TO2',
           'TH2', 'LevelO2', 'LevelH2', 'O2inH2', 'H2inO2', '脱氧上温', '脱氧下温', 'B塔上温', 'B塔下温',
           'C塔上温', 'C塔下温', 'A塔上温', 'A塔下温', 'DwePoint', '微氧量', '出罐压力', '进罐温度', '进罐压力']

    for folder in files:#首先读取大文件夹中的序列，然后读取里面所有20s的数据
        subfolder = os.path.join(orifolder, folder)#this line is compulsory, and if missed, isfile function will not be able to return the desired result
        subfiles = os.listdir(subfolder)
        subfiles.sort()
        for file20s in subfiles:
            subpath = os.path.join(subfolder, file20s)
            if os.path.isfile(subpath):
                pd = pandas.read_excel(subpath)
                all_data_20s = all_data_20s.append(pd)
            if os.path.isdir(subpath):
                files2s = os.listdir(subpath)
                files2s.sort()
                for file2s in files2s:
                    pd2s = pandas.read_excel(os.path.join(subpath,file2s))
                    all_data_2s = all_data_2s.append(pd2s)

    print(all_data_2s.shape)
    all_data_2s.index = np.arange(all_data_2s.shape[0])
    all_data_20s.index = np.arange(all_data_20s.shape[0])
    all_data_20s.columns=OfficialColumns
    all_data_2s.columns=OfficialColumns
    #all_data_2s.to_csv("AllData-2s.csv")#这里这种统一在一个文件中的数据形式应该是不能再用了
    #all_data_20s.to_csv("AllData-20s.csv")


"""这里是20220525根据神经网络文件的需要进行原始数据的重新提取，这里完全不做任何变换"""
this_time = 0
if this_time == 1:
    sourcefolder = 'Data-original'
    files = os.listdir(sourcefolder)
    files.sort()
    OriginalColumns = ['时间', '电解电压', '电解电流', '产氢量', '产氢累计量', '碱液流量', '碱温',
                       '系统压力  ', '氧槽温', '氢槽温', '氧侧液位', '氢侧液位', '氧中氢', '氢中氧', '脱氧上温', '脱氧下温',
                       'B塔上温', 'B塔下温', 'C塔上温', 'C塔下温', 'A塔上温', 'A塔下温', '露点', '微氧量', '出罐压力',
                       '进罐温度', '进罐压力']
    for folder in files:
        subfolder = os.path.join(sourcefolder,folder)
        print(subfolder)
        subfiles = os.listdir(subfolder)
        subfiles.sort()
        df = pandas.DataFrame()
        for file20s in subfiles:
            cur_path = os.path.join(subfolder,file20s)
            if os.path.isfile(cur_path):
                cur = pandas.read_excel(cur_path)
                df = pandas.concat([df,cur],axis=0)
        storage_file_name = 'Dynamic model data-20s/Data 0525/'+ folder+'.csv'
        'Data-original/TJ-20211129/TJ-20211129-083000-193000-20s.xls'
        df.to_csv(storage_file_name)
        print(storage_file_name)
        print(len(df))

