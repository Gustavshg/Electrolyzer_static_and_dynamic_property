"""这里开始的代码，我们希望能偶通过已有的极化曲线，来分析工作过程中电解槽偏离极化曲线的部分"""
import os
import time
import pandas
import Polar_fitting_collection as polar_collection
import Polar_data_loader  # 应该我们直接用这个文件就可以进行数据加载，各种日期的都可以
import matplotlib.pyplot as plt
t0 = time.time()
SourceFolder = "20s/Original"
SourceFiles = os.listdir(SourceFolder)
SourceFiles.sort()

loc_nn = []  # center
scale_nn = []  # standard deviation
loc_pc = []
scale_pc = []
date = []
for file in SourceFiles:
    date.append(file[-8:-4])
    current_file = SourceFolder + '/' + file
    polar_nn = polar_collection.polar_nn()

    data_0924 = Polar_data_loader.Original_Data_Loader(SourceFile='2s/Original/TJ-20211125.csv')
    inputs, voltage = data_0924.get_all_data()
    T_out, T_in, current_density, LyeFlow = data_0924.get_polar_data()

    polar = polar_nn.predict(inputs)
    error = polar - voltage
    a,b = polar_collection.error_N_analysis(error)
    loc_nn.append(a)
    scale_nn.append(b)
    polar = polar_collection.polar_shg(T_out, T_in, current_density, LyeFlow)
    error = polar - voltage
    a,b = polar_collection.error_N_analysis(error)
    loc_pc.append(a)
    scale_pc.append(b)


df = pandas.DataFrame()
df['date'] = date
df['loc nn'] = loc_nn
df['loc pc'] = loc_pc
df['scale nn'] = scale_nn
df['scale pc'] = scale_pc
df.to_csv('Norm distribution 0425.csv')


import numpy as np
x = np.arange(len(date))
total_width, n = 0.4,2
width = total_width / n
x = x - (total_width - width) / 2


plt.figure(figsize=(15, 8))
plt.bar(x,loc_nn)
plt.bar(x + width, loc_pc)
plt.legend(['NN','PolarCurve'])

print(time.time() - t0)

plt.show()


# plt.figure(figsize=(15, 8))
# ax1 = plt.gca()
# ax1.plot(voltage,alpha = 0.6)
# ax1.plot(polar)
# ax1.legend(['original', 'fitted'])
# ax1.set_ylabel('voltage')
# ax1.set_xlabel('Time')
# ax1.set_ylim([0,2.2])
#
# ax2 = ax1.twinx()
# ax2.set_ylabel('voltage error')
#
# ax2.plot([0] * len(polar_nn.predict(inputs)), color='r', alpha=0.7)
# ax2.scatter(range(len(polar_nn.predict(inputs))), error, alpha=0.3)
# ax2.set_ylim([-0.1,0.1])
#
# plt.title('recovered voltage-neural networks-0924-2s')
# plt.subplots_adjust(left=0.057, bottom=0.062, right=0.938, top=0.95)
#
# plt.figure(figsize=(8, 4))
# import seaborn as sns
#
# sns.set_style('darkgrid')
# sns.distplot(error,bins = 1000)
# plt.subplots_adjust(left=0.073, bottom=0.062, right=0.95, top=0.925)
# plt.title('loss comparison between polar curves')
# plt.xlim([-0.4, 0.4])
# print(error)
# plt.show()
