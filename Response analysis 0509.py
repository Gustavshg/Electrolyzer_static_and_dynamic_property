'''主要分析1130的数据'''
import pandas
import Polar_fitting_collection as polar_collection
import Polar_data_loader  # 应该我们直接用这个文件就可以进行数据加载，各种日期的都可以
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

plt.style.use('seaborn')

'''前四个循环'''
# nn_polar = polar_collection.polar_nn()
# data = Polar_data_loader.Original_Data_Loader(SourceFile='2s/Original/TJ-20211130.csv')
# inputs, voltage = data.get_all_data()
# T_out, T_in, current_density, LyeFlow = data.get_polar_data()
# step_seq = [3628, 5435, 7235, 9046, 10796, 12604, 12754, 14418, 16268, 44105, 45917, 47720, 49521, 51335, 53117, 54918,
#             56728, 58787]
# # polar_res = nn_polar.predict(inputs)
# polar_res = polar_collection.polar_shg(T_out, T_in, current_density, LyeFlow)
# # polar_res = polar_collection.polar_wtt(T_out,current_density)
#
#
# plt.figure(figsize=(15, 8))
# plt.title('1130-first-four')
# plt.subplots_adjust(left=0.073, bottom=0.062, right=0.95, top=0.925)
# bias = 0.02  # 绘制误差图时的每条基准线的偏差
#
# cd_dict = {0: 3500, 1: 2800, 2: 2100, 3: 1400}
# star_dict = {1: 447, 2: 599, 3: 744, 4: 893, 5: 1050, 6: 1195, 7: 1345}
# # end_dict = {0:534,1:827,2:1122,3:1423,4:1708}#这时候只选取前面的，不在每个阶跃中单独选取bias，所以这部分不需要了
# step_direct_dict = {1: 'down', 2: 'up', 3: 'down', 4: 'up', 5: 'down', 6: 'up', 7: 'down'}
# Step = 700
# StepTime = 300  # 每300秒变化一次电流
#
# Amp = pandas.DataFrame(
#     columns=['position', 'CurrentDensity', 'Step', 'StepDirection', 'error', 'error_corrected', 'StepTime'])
#
# for i in range(len(step_seq[:4])):#第几个电流密度
#     print(cd_dict[i])
#     i += 0  # 从第几个开始的
#     plt.xlim([0, 1750])
#     step_cur = step_seq[i]
#     step_next = step_seq[i + 1]
#
#     vol_cur = voltage[step_cur:step_next]
#     plt.plot(vol_cur, 'o')
#     vol_nn = polar_res[step_cur:step_next]
#     plt.plot(vol_nn)
#     # plt.plot(vol_cur - vol_nn + bias*i,'o')
#     # plt.plot([0,1770],[bias*i,bias*i], 'r')
#     err = np.array(vol_cur - vol_nn)
#     # err_dict[int(round(float(current_density[step_seq[i]]/50))*50)] = err[0:51]#这里是分析极化曲线时候的操作，现在肯定是不合适了
#
#     bias_end = np.average(err[1700:1750])  # 这部分是每个振荡周期最后一部分，作为bias可以用来平衡各部分的差异
#
#     record_length = 50
#     for k in [1, 2, 3, 4, 5, 6, 7]:#第几个阶跃
#         for j in range(record_length):#第几个采样点
#             start = star_dict[k] + j
#             cur = [[k,cd_dict[i],Step,step_direct_dict[k],float(err[start]),float(err[start]) - bias_end,StepTime]]
#             Cur = pandas.DataFrame(cur)
#             Cur.columns = [ 'position',  'CurrentDensity',            'Step',         'StepDirection',           'error', 'error_corrected',              'StepTime']
#             Amp = pandas.concat([Amp,Cur], ignore_index=True)
#             # Amp = Amp.append(cur, ignore_index=True)
#
#
# #Amp.to_csv('2s/DynamicResponse/amplitude_1130_first_four.csv')

# df = pandas.read_csv('2s/DynamicResponse/amplitude_1130_first_four.csv')
#
# plt.figure(figsize=(15, 8))
# plt.title('1130-first-four')
# plt.subplots_adjust(left=0.073, bottom=0.062, right=0.95, top=0.925)
# sns.violinplot(x = 'CurrentDensity', y = 'error_corrected', data = df, hue = 'StepDirection')
#
# plt.figure(figsize=(15, 8))
# ec = df['error_corrected']
# ec = abs(ec)
# df['error_corrected'] = ec
# plt.title('1130-first-four')
# plt.subplots_adjust(left=0.073, bottom=0.062, right=0.95, top=0.925)
# sns.barplot(x = 'CurrentDensity', y = 'error_corrected', data = df, hue = 'Step')
#
# plt.figure(figsize=(15, 8))
# plt.title('1130-first-four')
# plt.subplots_adjust(left=0.073, bottom=0.062, right=0.95, top=0.925)
# sns.barplot(x = 'CurrentDensity', y = 'error_corrected', data = df, hue = 'position')
'''中间四个循环'''

# nn_polar = polar_collection.polar_nn()
# data = Polar_data_loader.Original_Data_Loader(SourceFile='2s/Original/TJ-20211130.csv')
# inputs, voltage = data.get_all_data()
# T_out, T_in, current_density, LyeFlow = data.get_polar_data()
# step_seq = [ 44115, 45917, 47720, 49521, 51335]
# # polar_res = nn_polar.predict(inputs)
# polar_res = polar_collection.polar_shg(T_out, T_in, current_density, LyeFlow)
# # polar_res = polar_collection.polar_wtt(T_out,current_density)
#
#
# plt.figure(figsize=(15, 8))
# plt.title('1130-second-four')
# plt.subplots_adjust(left=0.073, bottom=0.062, right=0.95, top=0.925)
# bias = 0.02  # 绘制误差图时的每条基准线的偏差
#
# cd_dict = {0: 3500, 1: 2800, 2: 2100, 3: 1400}
# star_dict = {1: 349, 2: 406, 3: 446, 4: 527, 5: 590, 6: 642, 7: 710,8: 766,9: 822, 10: 884, 11: 948,12: 1004,13: 1066,14: 1124,15: 1184,16: 1246,17: 1306,18: 1362,19: 1424}
# # end_dict = {0:534,1:827,2:1122,3:1423,4:1708}#这时候只选取前面的，不在每个阶跃中单独选取bias，所以这部分不需要了
# step_direct_dict = {}
# for i in range(1,20):
#     if i % 2 == 1:
#         step_direct_dict[i] = 'down'
#     else:
#         step_direct_dict[i] = 'up'
# Step = 400
# StepTime = 120  # 每300秒变化一次电流
#
# Amp = pandas.DataFrame(
#     columns=['position', 'CurrentDensity', 'Step', 'StepDirection', 'error', 'error_corrected', 'StepTime'])
#
# for i in range(len(step_seq[:4])):#第几个电流密度
#     print(cd_dict[i])
#     i += 0  # 从第几个开始的
#     plt.xlim([0, 1750])
#     step_cur = step_seq[i]
#     step_next = step_seq[i + 1]
#
#     vol_cur = voltage[step_cur:step_next]
#     plt.plot(vol_cur, 'o')
#     vol_nn = polar_res[step_cur:step_next]
#     plt.plot(vol_nn)
#     # plt.plot(vol_cur - vol_nn + bias*i,'o')
#     # plt.plot([0,1770],[bias*i,bias*i], 'r')
#     err = np.array(vol_cur - vol_nn)
#
#     bias_end = np.average(err[1520:1600]) # 这部分是每个振荡周期最后一部分，作为bias可以用来平衡各部分的差异
#     record_length = 35
#     for k in range(1,20):#第几个阶跃
#         for j in range(record_length):#第几个采样点
#             start = star_dict[k] + j
#             cur = [[k,cd_dict[i],Step,step_direct_dict[k],float(err[start]),float(err[start]) - bias_end,StepTime]]
#             Cur = pandas.DataFrame(cur)
#             Cur.columns = [ 'position',  'CurrentDensity',            'Step',         'StepDirection',           'error', 'error_corrected',              'StepTime']
#             Amp = pandas.concat([Amp,Cur], ignore_index=True)
#
# # Amp.to_csv('2s/DynamicResponse/amplitude_1130_second_four.csv')
# #
# # df = pandas.read_csv('2s/DynamicResponse/amplitude_1130_second_four.csv')
#
#
#
# plt.figure(figsize=(15, 8))
# plt.title('1130-second-four')
# plt.subplots_adjust(left=0.073, bottom=0.062, right=0.95, top=0.925)
# sns.barplot(x = 'CurrentDensity', y = 'error_corrected', data = df, hue = 'StepDirection')
#
# plt.figure(figsize=(15, 8))
# ec = df['error_corrected']
# ec = abs(ec)
# df['error_corrected'] = ec
# plt.title('1130-second-four')
# plt.subplots_adjust(left=0.073, bottom=0.062, right=0.95, top=0.925)
# sns.barplot(x = 'CurrentDensity', y = 'error_corrected', data = df, hue = 'Step')
#
# plt.figure(figsize=(15, 8))
# plt.title('1130-second-four')
# plt.subplots_adjust(left=0.073, bottom=0.062, right=0.95, top=0.925)
# sns.barplot(x = 'CurrentDensity', y = 'error_corrected', data = df, hue = 'position')

'''最后四个循环'''
# nn_polar = polar_collection.polar_nn()
# data = Polar_data_loader.Original_Data_Loader(SourceFile='2s/Original/TJ-20211130.csv')
# inputs, voltage = data.get_all_data()
# T_out, T_in, current_density, LyeFlow = data.get_polar_data()
# step_seq = [ 51315, 53112, 54922, 56718, 58787]#每个分段的开始
# # polar_res = nn_polar.predict(inputs)
# polar_res = polar_collection.polar_shg(T_out, T_in, current_density, LyeFlow)
# # polar_res = polar_collection.polar_wtt(T_out,current_density)
#
#
# plt.figure(figsize=(15, 8))
# plt.title('1130-last-four')
# plt.subplots_adjust(left=0.073, bottom=0.062, right=0.95, top=0.925)
# bias = 0.02  # 绘制误差图时的每条基准线的偏差
#
# cd_dict = {0: 3500, 1: 2800, 2: 2100, 3: 1400}#电流密度
# star_dict = {1: 358, 2: 420, 3: 481, 4: 540, 5: 600, 6: 654, 7: 719,8: 779,9: 839, 10: 899, 11: 960,12: 1022,13: 1080,14: 1138,15: 1202,16: 1261,17: 1324,18: 1380,19: 1440}#每个小段开始
# # end_dict = {0:534,1:827,2:1122,3:1423,4:1708}#这时候只选取前面的，不在每个阶跃中单独选取bias，所以这部分不需要了
# step_direct_dict = {}
# for i in range(1,20):
#     if i % 2 == 1:
#         step_direct_dict[i] = 'down'
#     else:
#         step_direct_dict[i] = 'up'
# Step = 700
# StepTime = 120  # 每300秒变化一次电流
#
# Amp = pandas.DataFrame(
#     columns=['position', 'CurrentDensity', 'Step', 'StepDirection', 'error', 'error_corrected', 'StepTime'])
#
# for i in range(len(step_seq[:4])):#第几个电流密度
#     print(cd_dict[i])
#     i += 0  # 从第几个开始的
#     plt.xlim([0, 1750])
#     step_cur = step_seq[i]
#     step_next = step_seq[i + 1]
#
#     vol_cur = voltage[step_cur:step_next]
#     plt.plot(vol_cur, 'o')
#     vol_nn = polar_res[step_cur:step_next]
#     plt.plot(vol_nn)
#     # plt.plot(vol_cur - vol_nn + bias*i,'o')
#     # plt.plot([0,1770],[bias*i,bias*i], 'r')
#     err = np.array(vol_cur - vol_nn)
#
#     bias_end = np.average(err[1550:1600]) # 这部分是每个振荡周期最后一部分，作为bias可以用来平衡各部分的差异
#     record_length = 35
#     for k in range(1,20):#第几个阶跃
#         for j in range(record_length):#第几个采样点
#             start = star_dict[k] + j
#             cur = [[k,cd_dict[i],Step,step_direct_dict[k],float(err[start]),float(err[start]) - bias_end,StepTime]]
#             Cur = pandas.DataFrame(cur)
#             Cur.columns = [ 'position',  'CurrentDensity',            'Step',         'StepDirection',           'error', 'error_corrected',              'StepTime']
#             Amp = pandas.concat([Amp,Cur], ignore_index=True)
#
# Amp.to_csv('2s/DynamicResponse/amplitude_1130_last_four.csv')
#
# df = pandas.read_csv('2s/DynamicResponse/amplitude_1130_last_four.csv')
#
#
#
# plt.figure(figsize=(15, 8))
# plt.title('1130-last-four')
# plt.subplots_adjust(left=0.073, bottom=0.062, right=0.95, top=0.925)
# sns.barplot(x = 'CurrentDensity', y = 'error_corrected', data = df, hue = 'StepDirection')
#
# plt.figure(figsize=(15, 8))
# ec = df['error_corrected']
# ec = abs(ec)
# df['error_corrected'] = ec
# plt.title('1130-last-four')
# plt.subplots_adjust(left=0.073, bottom=0.062, right=0.95, top=0.925)
# sns.barplot(x = 'CurrentDensity', y = 'error_corrected', data = df, hue = 'Step')
#
# plt.figure(figsize=(15, 8))
# plt.title('1130-last-four')
# plt.subplots_adjust(left=0.073, bottom=0.062, right=0.95, top=0.925)
# sns.barplot(x = 'CurrentDensity', y = 'error_corrected', data = df, hue = 'position')

'''合并考虑'''
df_1 = pandas.read_csv('2s/DynamicResponse/amplitude_1129_all.csv')
df_2 = pandas.read_csv('2s/DynamicResponse/amplitude_1130_all.csv')

df = pandas.concat([df_1,df_2],ignore_index=True)

plt.figure(figsize=(15, 8))
plt.title('1129+1130 two direction, step = 700A/m2')
plt.subplots_adjust(left=0.073, bottom=0.062, right=0.95, top=0.925)
sns.barplot(x = 'CurrentDensity', y = 'error_corrected', data = df[df['Step'] == 700], hue = 'StepDirection')

plt.figure(figsize=(15, 8))
ec = df['error_corrected']
ec = abs(ec)
df['error_corrected'] = ec
plt.title('1129+1130 abs value')
plt.subplots_adjust(left=0.073, bottom=0.062, right=0.95, top=0.925)
sns.barplot(x = 'CurrentDensity', y = 'error_corrected', data = df[df['StepDirection'] == 'up'], hue = 'Step')

# plt.figure(figsize=(15, 8))
# plt.title('1129+1130 sequence')
# plt.subplots_adjust(left=0.073, bottom=0.062, right=0.95, top=0.925)
# sns.barplot(x = 'CurrentDensity', y = 'error_corrected', data = df, hue = 'position')

plt.figure(figsize=(15, 8))
plt.title('1129+1130 step time, step = 700A/m2')
plt.subplots_adjust(left=0.073, bottom=0.062, right=0.95, top=0.925)
sns.barplot(x = 'CurrentDensity', y = 'error_corrected', data = df[df['Step'] == 700], hue = 'StepTime')

# plt.figure(figsize=(8, 8))
l = ['CurrentDensity','StepDirection','Step','StepTime']
sns.pairplot(df[l])
# plt.title('1129+1130 step time, step = 700A/m2')
plt.subplots_adjust(left=0.073, bottom=0.062, right=0.95, top=0.925)


''''''
plt.show()
