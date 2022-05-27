"""这里主要是根据神经网络模型的注意事项，准备对数据进行校验，看看有没有异常，并且把初步分析之后的数据附加到每个文件之后"""
import os
import pandas
import numpy as np
import matplotlib.pyplot as plt

# 引入matplotlib字體管理 FontProperties
from matplotlib.font_manager import FontProperties

# 設置我們需要用到的中文字體（字體文件地址）
my_font = FontProperties(fname=r"c:\windows\fonts\SimHei.ttf", size=12)




OriginalColumns = ['时间', '电解电压', '电解电流', '产氢量', '产氢累计量',
                   '碱液流量', '碱温', '系统压力  ', '氧槽温', '氢槽温', '氧侧液位', '氢侧液位', '氧中氢', '氢中氧',
                   '脱氧上温', '脱氧下温', 'B塔上温', 'B塔下温', 'C塔上温', 'C塔下温', 'A塔上温', 'A塔下温', '露点',
                   '微氧量', '出罐压力', '进罐温度', '进罐压力']
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


source_folder = 'Dynamic model data-20s/Data 0525'
dates = os.listdir(source_folder)
dates.sort()

"""这部分主要是检查我们需要的列中是否存在-9999这样的异常值，应该只有1202里面有这样的异常值"""
columns_exam = 0
if columns_exam == 1:
    # exam_columns = ['电解电压', '电解电流', '碱液流量', '碱温', '系统压力  ', '氧槽温', '氢槽温', '氧中氢', '氢中氧']
    exam_columns = [ '氧槽温', '氢槽温']
    for date in dates:
        flag = 1  # 如果没有数据异常，就为1，如果有数据异常，就置为0
        anomaly = []
        df = pandas.read_csv(os.path.join(source_folder, date))
        for column in exam_columns:
            cur_data = df[column]
            if min(cur_data) <= 0:
                flag = 0
                anomaly.append(column)

        if flag == 1:
            print(date + ' all clear')
        else:
            # slices = [1465,1541]#1130的数据里面有两个点全部数据为零，这里需要对其进行一个差值
            # for slice in slices:
            #     for col in OriginalColumns[1:]:
            #         # print(col)
            #         # print(cur[slice])
            #         df[col][slice] = (df[col][slice-1] + df[col][slice+1])/2
            #     # print(col)
            #     print(df[ '氧槽温'][slice])
            # df.to_csv(os.path.join(source_folder, date))
            print(date + str(anomaly) + 'has anomaly')

"""这部分主要是根据需要新增需要的列"""
add_columns = 0
if add_columns == 1:
    for date in dates:
        df = pandas.read_csv(os.path.join(source_folder, date))
        """模型输入"""
        col = 'V'  # 小室电压
        if not col in df:
            df[col] = df['电解电压'] / 34
        col = 'I'  # 总电流
        if not col in df:
            df[col] = df['电解电流']
        col = 'Current density'  # 电流密度
        if not col in df:
            df[col] = df['电解电流'] / 0.425
        col = 'Pressure'  # 制氢压力
        if not col in df:
            df[col] = df['系统压力  ']
        col = 'Tlye'  # 碱液温度
        if not col in df:
            df[col] = df['碱温']
        col = 'TH2'  # 氢槽温
        if not col in df:
            df[col] = df['氢槽温']
        col = 'TO2'  # 氧槽温
        if not col in df:
            df[col] = df['氧槽温']
        col = 'Tout'  # 出口平均温度
        if not col in df:
            df[col] = (df['氢槽温'] + df['氧槽温']) / 2
        col = 'LyeFlow'  # 碱液流量（动态模型使用）
        if not col in df:
            df[col] = df['碱液流量']
        col = 'LyeFlow_Polar'  # 碱液流量（极化曲线使用）
        if not col in df:
            cur = df['碱液流量']
            for i in range(len(cur)):
                if cur[i] <= 0.204425:
                    cur[i] = 0.204425
            df[col] = cur
        col = 'dI'  # 当前时刻的电流与上一时刻电流的差异，所以第一个是0
        if not col in df:
            dI = [0]
            cur = df['电解电流']
            for i in range(1, len(cur)):
                dI.append(cur[i] - cur[i - 1])
            df[col] = dI
        col = 'dj'  # 当前时刻的电流与上一时刻  电流密度  的差异，所以第一个是0
        if not col in df:
            df[col] = df['dI'] / 0.425

        """模型标的"""
        col = 'dV'  # 当前时刻的电流与上一时刻 小室电压 的差异，所以第一个是0
        if not col in df:
            dV = [0]
            cur = df['V']
            for i in range(1, len(cur)):
                dV.append(cur[i] - cur[i - 1])
            df[col] = dV

        col = 'dTout'  # 当前时刻的电流与上一时刻出口平均温度的差异，所以第一个是0
        if not col in df:
            dTout = [0]
            cur = df['Tout']
            for i in range(1, len(cur)):
                dTout.append(cur[i] - cur[i - 1])
            df[col] = dTout

        col = 'HTO'  # 氧中氢，hydrogen to oxygen
        if not col in df:
            df[col] = df['氧中氢']
        col = 'OTH'  # 氢中氧，oxygen to hydrogen
        if not col in df:
            df[col] = df['氢中氧']
        print(date, df.columns)
        df.to_csv(os.path.join(source_folder, date))

    # plt.style.use('seaborn')
    # plt.figure(figsize=(15, 8))
    # plt.title('当前时刻电流与上一时刻的差值',fontproperties=my_font)
    # plt.subplots_adjust(left=0.073, bottom=0.062, right=0.95, top=0.925)
    # plt.plot(df['dI'])
    # plt.show()

"""加入极化曲线电压"""

import time

OriginalColumns = ['时间', '电解电压', '电解电流', '产氢量', '产氢累计量',
                   '碱液流量', '碱温', '系统压力  ', '氧槽温', '氢槽温', '氧侧液位', '氢侧液位', '氧中氢', '氢中氧',
                   '脱氧上温', '脱氧下温', 'B塔上温', 'B塔下温', 'C塔上温', 'C塔下温', 'A塔上温', 'A塔下温', '露点',
                   '微氧量', '出罐压力', '进罐温度', '进罐压力', 'V', 'I', 'Current density', 'Pressure',
                   'Tlye', 'TH2', 'TO2', 'Tout', 'LyeFlow', 'LyeFlow_Polar', 'dI', 'dj',
                   'dV', 'dTout', 'HTO', 'OTH'],
t0 = time.time()
add_polar = 0
if add_polar == 1:
    import Polar_fitting_collection as polar
    """这里需要把电流密度、入口温度、碱液流量进行错位，用下一时刻的这三个数值，再加上当前时刻的出口温度，得到当前时刻预测下一时刻的极化电压"""
    nn_polar = polar.polar_nn()
    for date in dates:
        print(date)
        df = pandas.read_csv(os.path.join(source_folder, date))
        T_out = df['Tout']
        current_density = df['Current density']
        T_in = df['Tlye']
        LyeFlow = df['LyeFlow_Polar']
        V_nn = nn_polar.polar(T_out, T_in, current_density, LyeFlow)
        V_lh = polar.polar_lihao(T_out,current_density)
        V_wtt = polar.polar_wtt(T_out,current_density)
        V_shg = polar.polar_shg(T_out,T_in,current_density,LyeFlow)
        df['polar_nn'] = V_nn
        df['polar_lh'] = V_lh
        df['polar_wtt'] = V_wtt
        df['polar_shg'] = V_shg
        df.to_csv(os.path.join(source_folder, date))

    print(time.time()- t0)

add_static_and_dynamic_voltage = 0
if add_static_and_dynamic_voltage == 1:
    import Polar_fitting_collection as polar
    """这里根据我们电化学动态响应的思路进行改造，即使用上一时刻温度与当前时刻电流密度、碱液流量、入口温度等计算当前时刻静态电压，并且和动态电压做差值，得到模型预测的标的"""
    nn_polar = polar.polar_nn()
    for date in dates:
        print(date)
        df = pandas.read_csv(os.path.join(source_folder, date))
        print(df.columns)
        T_out = df['Tout']

        T_out = list(T_out[1:])
        T_out.append(T_out[-1])
        T_out = np.array(T_out)
        current_density = df['Current density']
        T_in = df['Tlye']
        LyeFlow = df['LyeFlow_Polar']
        V_static = nn_polar.polar(T_out, T_in, current_density, LyeFlow)

        df['V_static'] = V_static#静态电压即为预测下一状态的极化电压
        df['V_dynamic'] = df['V']-V_static#动态电压就是下一秒真实值和静态电压的差值
        df.to_csv(os.path.join(source_folder, date))

        # ax1 = plt.gca()
        # ax1.plot(df['V_static'],'green')
        # ax1.plot(df['V'],'red')
        # ax1.legend(['static voltage','original voltage'],loc = 1)
        # ax2 = ax1.twinx()
        # ax2.plot(df['V_dynamic'])
        # ax2.legend(['dynamic voltage'], loc = 2)
        # plt.show()

    print(time.time()- t0)




smooth= 0
if smooth == 1:
    import Smoothen as sm

    for date in dates:
        print(date)
        df = pandas.read_csv(os.path.join(source_folder, date))
        Tout = df['Tout']
        dT = df['dTout']
        if not 'dTout_WL' in df:
            # Tout_AA = sm.AA(Tout)
            # Tout_EMA = sm.EMA(Tout)
            Tout_WL = sm.WL(Tout)

            # dT_AA = sm.diff(Tout_AA)
            # dT_EMA = sm.diff(Tout_EMA)
            dT_WL = sm.diff(Tout_WL)
            df['dTout_WL'] = dT_WL
            df.to_csv(os.path.join(source_folder, date))

    # step = 0.3
    # plt.style.use('seaborn')
    # plt.figure(figsize=(15, 8))
    # plt.title('不同平滑方式与效果',fontproperties=my_font)
    # plt.subplots_adjust(left=0.073, bottom=0.062, right=0.95, top=0.925)
    # plt.plot(df['dTout'],label = 'raw')
    # plt.plot(dT_AA + step, label='algebra average')
    # plt.plot(dT_EMA + step*2, label='moving average')
    # plt.plot(dT_WL + step*3, label='WL transform')
    # plt.legend()
    # plt.show()
