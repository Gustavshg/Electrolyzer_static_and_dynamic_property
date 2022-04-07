#这个程序是针对2021年11月底在新乡开展的电解槽测试的数据处理及最后模型搭建的问题
#最终的目的是想建立包含电解槽各种动态特性响应的数据驱动模型
import os
import pandas
import numpy as np
import xlrd
import matplotlib
import openpyxl

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
#all_data_2s.to_csv("AllData-2s.csv")
#all_data_20s.to_csv("AllData-20s.csv")