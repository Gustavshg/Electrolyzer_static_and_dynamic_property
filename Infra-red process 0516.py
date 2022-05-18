import numpy as np
import pandas
import os
import Infra_red_image_analysis as img_analysis
import matplotlib.pyplot as plt
import time

t0 = time.time()

sourcefolder = 'Infra images/1125'
sourcefilelist = os.listdir(sourcefolder)
sourcefilelist.sort()
'''以下是读取数据'''
read = 1
if read == 1:
    i = 0
    slope_seq = []
    intercept_seq = []
    for f in sourcefilelist:
        file = sourcefolder + '/' + f
        print(i,file)
        img = img_analysis.read_infra(file)
        data = img_analysis.temp_distribution(img)
        bars = [0, 1, 8, 9, 11]
        temp_min = min(data[1])
        temp_max = max(data[1])
        thre = 0.6 * temp_min + 0.4 * temp_max
        thre_max = 50
        if thre >= thre_max:
            thre = thre_max
        seq, temp_big, temp_small, height_big, height_small, width_big, width_small, slope, intercept = img_analysis.temp_slope_analysis(data,thre,bars = bars)


        slope_seq.append(slope)
        intercept_seq.append(intercept)

        i += 1
    slope_seq = np.abs(slope_seq)

    slope_seq = np.expand_dims(slope_seq,axis=1)
    intercept_seq = np.expand_dims(intercept_seq,axis=1)
    df_slope_intercept = np.concatenate((slope_seq,intercept_seq),axis = 1)
    df_slope_intercept = pandas.DataFrame(df_slope_intercept)
    df_slope_intercept.columns = ['slope','intercept']
    df_slope_intercept.to_csv('Cache/slope and intercept 1125-2')

    print(i,time.time()-t0)
    plt.figure()
    plt.plot(slope_seq)
    plt.title('slope')
    plt.figure()
    plt.plot(intercept_seq)
    plt.title('intercept')
'''以下是分析数据'''

compare = 0
if compare == 1:
    df_1125 = pandas.read_csv('Cache/slope and intercept 1125-2')
    slope = df_1125['slope']
    intercept = df_1125['intercept']
    # plt.style.use('seaborn')
    plt.figure(figsize=(15, 8))
    plt.subplots_adjust(left=0.073, bottom=0.062, right=0.95, top=0.925)
    ax1 = plt.gca()
    ax1.plot(slope,label = 'slope',color = 'green')
    ax1.set_ylabel('slope')
    plt.legend(loc = 2)
    ax2 = ax1.twinx()
    ax2.plot(intercept,label = 'intercept')
    ax2.set_ylabel('intercept')
    plt.title('slope and intercept analysis of 1125')
    plt.legend()

    plt.figure(figsize=(15, 8))
    plt.subplots_adjust(left=0.073, bottom=0.062, right=0.95, top=0.925)
    df = pandas.read_csv('20s/Original/TJ-20211125.csv')
    TH = df['氢槽温']
    TO = df['氧槽温']
    Tlye = df['碱温']
    plt.plot(TH[93:2231],label = 'T H2')
    plt.plot(TO[93:2231],label = 'T O2')
    plt.plot(Tlye[93:2231],label = 'T lye')
    plt.figure(figsize=(15, 8))
    plt.subplots_adjust(left=0.073, bottom=0.062, right=0.95, top=0.925)
    cd = df['电解电流']
    plt.plot(cd[93:2231],label = 'currrent')
    plt.legend


    plt.title('original data of 1125')
    plt.legend()
'''以下是解剖分析1125数据的代码'''
exam = 0
if exam == 1:
    file = 'Infra images/1125/IMG20211125123836.txt'
    img = img_analysis.read_infra(file)

    data = img_analysis.temp_distribution(img)

    # img_analysis.img_show(img,file)
    img_analysis.show_violin_gaussian_slope(data,file,bars = [0,1,8,9,11])
    # img_analysis.exam(data,4,file)

plt.show()






