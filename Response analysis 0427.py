import os
import time
import pandas
import Polar_fitting_collection as polar_collection
import Polar_data_loader  # 应该我们直接用这个文件就可以进行数据加载，各种日期的都可以
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
#
# df1 = pandas.read_csv('2s/DynamicResponse/1202all.csv')
# df2 = pandas.read_csv('2s/DynamicResponse/1125+1126.csv')
# # df3 = pandas.read_csv('2s/DynamicResponse/1202-3.csv')
# df_all = df1.append(df2)
# # df_all = df_all.append(df3)
# df_all.index = range(len(df_all))
# print(df_all)
# df_all.to_csv('2s/DynamicResponse/all_polar_data.csv')

df = pandas.read_csv('2s/DynamicResponse/all_polar_data.csv')
cd = df['CurrentDensity']
for i in range(len(df)):
    if cd[i] == 1150:
        cd[i] = 1200
df['CurrentDensity'] = cd
# df.to_csv('2s/DynamicResponse/all_polar_data.csv')
plt.style.use('seaborn')
plt.figure(figsize=(15, 8))

sns.violinplot(x = 'CurrentDensity', y = 'error', data = df, hue = 'StepDirection')
plt.title('Step error of all polar data')
plt.subplots_adjust(left=0.073, bottom=0.062, right=0.95, top=0.925)
plt.show()


# plt.style.use('seaborn')
#
#
# t0 = time.time()
# nn_polar = polar_collection.polar_nn()
# data = Polar_data_loader.Original_Data_Loader(SourceFile='2s/Original/TJ-20211202.csv')
# inputs, voltage = data.get_all_data()
# T_out, T_in, current_density, LyeFlow = data.get_polar_data()
# step_seq =[2098, 2704, 3185, 3608, 4059, 4511, 4960, 5412, 5862, 6307, 6738, 8500, 9004, 9460, 9908, 10356, 10811,11257, 11714, 12161, 12605, 13000, 13501, 13968, 14407, 14856, 15310, 15757, 16208, 16657, 17107, 17540]
# polar_res = nn_polar.predict(inputs)
# # polar_res = polar_collection.polar_shg(T_out,T_in,current_density,LyeFlow)
# # polar_res = polar_collection.polar_wtt(T_out,current_density)
#
#
#
#
# plt.figure(figsize=(15, 8))
# plt.title('1202-first-down')
# plt.subplots_adjust(left=0.073, bottom=0.062, right=0.95, top=0.925)
# bias = 0.02
# err_dict = {}
# for i in range(len(step_seq[1:11])):
#     i+=0
#     vol_cur = voltage[step_seq[i]:step_seq[i]+400]
#     # plt.plot(vol_cur,'o')
#     vol_nn = polar_res[step_seq[i]:step_seq[i]+400]
#     # plt.plot(vol_nn)
#     plt.plot(vol_cur - vol_nn + bias*i,'o')
#     plt.plot([0,400],[bias*i,bias*i], 'r')
#     err = np.array(vol_cur-vol_nn)
#     err_dict[int(round(float(current_density[step_seq[i]+50]/50))*50)] = err[0:51]
#
# df = pandas.DataFrame(columns = ['CurrentDensity','StepDirection','error'])
# for cd in err_dict.keys():
#     cur = {}
#     cur['CurrentDensity'] = cd
#     cur['StepDirection'] = 'down'
#     for e in err_dict[cd]:
#         cur['error'] = e[0]
#         df = df.append(cur, ignore_index=True)
# print(df)
#
# StorageFile = '2s/DynamicResponse/1202-1.csv'
#
# df.to_csv(StorageFile)
#
#
# plt.figure(figsize=(15, 8))
# plt.title('1202-second-down')
# plt.subplots_adjust(left=0.073, bottom=0.062, right=0.95, top=0.925)
#
# err_dict = {}
# for i in range(len(step_seq[13:22])):
#     i +=11
#     vol_cur = voltage[step_seq[i]:step_seq[i]+400]
#     # plt.plot(vol_cur,'o')
#     vol_nn = polar_res[step_seq[i]:step_seq[i]+400]
#     # plt.plot(vol_nn)
#     plt.plot(vol_cur - vol_nn + bias*i,'o')
#     plt.plot([0,400],[bias*i,bias*i], 'r')
#     err = np.array(vol_cur-vol_nn)
#     err_dict[int(round(float(current_density[step_seq[i]+50]/50))*50)] = err[0:51]
#
#
# df = pandas.DataFrame(columns = ['CurrentDensity','StepDirection','error'])
# for cd in err_dict.keys():
#     cur = {}
#     cur['CurrentDensity'] = cd
#     cur['StepDirection'] = 'down'
#     for e in err_dict[cd]:
#         cur['error'] = e[0]
#         df = df.append(cur, ignore_index=True)
# print(df)
#
# StorageFile = '2s/DynamicResponse/1202-2.csv'
#
# df.to_csv(StorageFile)
#
#
#
#
#
# plt.figure(figsize=(15, 8))
# plt.title('1202-third-up')
# plt.subplots_adjust(left=0.073, bottom=0.062, right=0.95, top=0.925)
#
# err_dict = {}
# for i in range(len(step_seq[23:])):
#     i += 22
#     vol_cur = voltage[step_seq[i]:step_seq[i]+400]
#     # plt.plot(vol_cur,'o')
#     vol_nn = polar_res[step_seq[i]:step_seq[i]+400]
#     # plt.plot(vol_nn)
#     plt.plot(vol_cur - vol_nn + bias*i,'o')
#     plt.plot([0,400],[bias*i,bias*i], 'r')
#     err = np.array(vol_cur-vol_nn)
#     err_dict[int(round(float(current_density[step_seq[i]+50]/50))*50)] = err[0:51]
#
#
#
# df = pandas.DataFrame(columns = ['CurrentDensity','StepDirection','error'])
# for cd in err_dict.keys():
#     cur = {}
#     cur['CurrentDensity'] = cd
#     cur['StepDirection'] = 'up'
#     for e in err_dict[cd]:
#         cur['error'] = e[0]
#         df = df.append(cur, ignore_index=True)
# print(df)
#
# StorageFile = '2s/DynamicResponse/1202-3.csv'
#
# df.to_csv(StorageFile)
#
#
#
# # plt.figure(figsize=(15, 8))
# # plt.style.use('seaborn')
#
#
#
# # print(df)
# ## cur = {'CurrentDensity':10,'StepDirection':'up','error':0.1}
# #
# # for cd in err_dict.keys():
# #     cur = {}
# #     cur['CurrentDensity'] = cd
# #     cur['StepDirection'] = 'down'
# #     for e in err_dict[cd]:
# #         cur['error'] = e[0]
# #         df = df.append(cur, ignore_index=True)
# # print(df)
#
# # StorageFile = '2s/DynamicResponse/1125.csv'
#
# # df.to_csv(StorageFile)
#
# plt.show()



















# plt.figure(figsize=(15, 8))
# plt.style.use('seaborn')
# df = pandas.DataFrame(columns = ['CurrentDensity','StepDirection','error'])
# print(df)
# cur = {'CurrentDensity':10,'StepDirection':'up','error':0.1}
#
#
# for cd in err_dict.keys():
#     cur = {}
#     cur['CurrentDensity'] = cd
#     if cd == 950:
#         cur['StepDirection'] = 'down'
#     else:
#         cur['StepDirection'] = 'up'
#     for e in err_dict[cd]:
#         cur['error'] = e[0]
#         df = df.append(cur, ignore_index=True)
# print(df)
# StoreFile = '2s/DynamicResponse/1126.csv'
# df.to_csv(StoreFile)







# delta = 120
# plt.plot(current_density)
# step_seq =[]
# for i in range(2,len(current_density)-50):
#     if abs(current_density[i+5] - current_density[i]) > delta and abs(current_density[i] - current_density[i-1]) < delta:
#         if not step_seq or i-step_seq[-1] >10:
#     # if  i in [2098, 2704, 3185, 3608, 4059, 4511, 4960, 5412, 5862, 6307, 6738, 8500, 9004, 9460, 9908, 10356, 10811,11257, 11714, 12161, 12605, 13000, 13501, 13968, 14407, 14856, 15310, 15757, 16208, 16657, 17107, 17540]:
#             step_seq.append(i)
#             plt.scatter(i, current_density[i], color='r')
#             print(i)
# print(step_seq)
# title = 'Deviation between each sample time is' + str(delta)
# plt.legend(['Current Density','change points'])
# plt.title(title)
# plt.subplots_adjust(left=0.073, bottom=0.062, right=0.95, top=0.925)





# plt.figure(figsize=(15, 8))
# plt.style.use('seaborn')
# delta_seq = []
# for i in range(1,len(current_density)-1):
#     delta_seq.append(abs(current_density[i] - current_density[i-1]))
# plt.scatter(range(len(delta_seq)),delta_seq)
# plt.plot([0,len(delta_seq)],[1,1],'r')
# plt.plot([0,len(delta_seq)],[10,10],'r')
# plt.plot([0,len(delta_seq)],[100,100],'r')
# plt.yscale("log")
# plt.title('Current density change in time sequence')
# plt.subplots_adjust(left=0.073, bottom=0.062, right=0.95, top=0.925)


#
# print(inputs[50:100])
# print(nn_polar.polar(13.,55.,0.0004,0.204425))
#
# plt.style.use('seaborn')
# t0 = time.time()
#
# polar = nn_polar.polar(T_out,T_in,current_density,LyeFlow)
#
#
# print(time.time()-t0)
# plt.plot(voltage)
# plt.plot(polar)
# plt.subplots_adjust(left=0.073, bottom=0.062, right=0.95, top=0.925)
# plt.legend(['original','neural network'])
# plt.title('comparison of 1125')
# plt.show()
