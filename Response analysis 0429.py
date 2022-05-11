import time
import pandas
import Polar_fitting_collection as polar_collection
import Polar_data_loader  # 应该我们直接用这个文件就可以进行数据加载，各种日期的都可以
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns




plt.style.use('seaborn')


"""1129"""
# t0 = time.time()
# nn_polar = polar_collection.polar_nn()
# data = Polar_data_loader.Original_Data_Loader(SourceFile='2s/Original/TJ-20211129.csv')
# inputs, voltage = data.get_all_data()
# T_out, T_in, current_density, LyeFlow = data.get_polar_data()
# step_seq =[3935, 5741, 7550, 9368,  11784, 13671,15470 ]
# polar_res = nn_polar.predict(inputs)
# polar_res = polar_collection.polar_shg(T_out,T_in,current_density,LyeFlow)
# polar_res = polar_collection.polar_wtt(T_out,current_density)


# plt.figure(figsize=(15, 8))
# plt.title('1129-first-small-step')
# plt.subplots_adjust(left=0.073, bottom=0.062, right=0.95, top=0.925)
# plt.xlim([25,1760])
# bias = 0.02#绘制误差图时的每条基准线的偏差
# cd_dict = {0:1400,1:2100,2:2800,3:3500}
# star_dict = {0:300,1:613,2:911,3:1218,4:1515}
# end_dict = {0:534,1:827,2:1122,3:1423,4:1708}
# Amp = pandas.DataFrame(columns = ['CurrentDensity','Step','StepDirection','error'])
# Bias = pandas.DataFrame(columns = ['CurrentDensity','Step','StepDirection','error'])
# for i in range(len(step_seq[:4])):
#     step_cur = step_seq[i]
#     step_next = step_seq[i+1]
#     i+=0
#     vol_cur = voltage[step_cur:step_next]
#     plt.plot(vol_cur,'o')
#     vol_nn = polar_res[step_cur:step_next]
#     plt.plot(vol_nn)
#     # plt.plot(vol_cur - vol_nn + bias*i,'o')
#     # plt.plot([0,1770],[bias*i,bias*i], 'r')
#     err = np.array(vol_cur-vol_nn)
#     # err_dict[int(round(float(current_density[step_seq[i]]/50))*50)] = err[0:51]#这里是分析极化曲线时候的操作，现在肯定是不合适了
#     record_length = 50
#     for k in [0,1,2,3,4]:
#         for j in range(record_length):
#             start = star_dict[k] + j
#             end = end_dict[k] + j
#             cur = {}
#             cur['CurrentDensity'] = cd_dict[i]
#             if k == 0:
#                 cur['Step'] = '350'
#                 cur['StepDirection'] = 'up'
#             if k ==1:
#                 cur['Step'] = '700'
#                 cur['StepDirection'] = 'down'
#             if k ==2:
#                 cur['Step'] = '700'
#                 cur['StepDirection'] = 'up'
#             if k == 3:
#                 cur['Step'] = '700'
#                 cur['StepDirection'] = 'down'
#             if k == 4:
#                 cur['Step'] = '350'
#                 cur['StepDirection'] = 'up'
#             cur['error'] = float(err[start])
#             Amp = Amp.append(cur,ignore_index=True)
#             cur['error'] = float(err[end])
#             Bias = Bias.append(cur,ignore_index=True)

# Amp.to_csv('2s/DynamicResponse/amplitude_1129_first_three.csv')
# Bias.to_csv('2s/DynamicResponse/bias_1129_first_three.csv')
# print(Amp)
# print(Bias)

# df_amp = pandas.read_csv('2s/DynamicResponse/amplitude_1129_first_three.csv')
# df_bias = pandas.read_csv('2s/DynamicResponse/bias_1129_first_three.csv')
#
# # ec = df_amp['error_corrected']
# # ec *=-1
# # df_amp['error_corrected'] = ec
# # df_amp.to_csv('2s/DynamicResponse/amplitude_1129_first_three.csv')
# plt.figure(figsize=(15, 8))
#
# sns.violinplot(x = 'CurrentDensity', y = 'error_corrected', data = df_amp, hue = 'StepDirection')
# # sns.boxplot(x = 'CurrentDensity', y = 'error', data = df, hue = 'Step')
# plt.title('Step error of 1129 first')
# plt.subplots_adjust(left=0.073, bottom=0.062, right=0.95, top=0.925)
#
#
# l = ['CurrentDensity','error_corrected','StepDirection','Step']
# sns.pairplot(df_amp[l])



#1129第二部分
# t0 = time.time()
nn_polar = polar_collection.polar_nn()
data = Polar_data_loader.Original_Data_Loader(SourceFile='2s/Original/TJ-20211129.csv')
inputs, voltage = data.get_all_data()
T_out, T_in, current_density, LyeFlow = data.get_polar_data()
step_seq =[3935, 5741, 7550, 9368,  11765, 13671,15490 ]
polar_res = nn_polar.predict(inputs)

# polar_res = polar_collection.polar_shg(T_out,T_in,current_density,LyeFlow)
polar_res = polar_collection.polar_wtt(T_out,current_density)


plt.figure(figsize=(15, 8))
plt.title('1129-last-large-step')
plt.subplots_adjust(left=0.073, bottom=0.062, right=0.95, top=0.925)
plt.xlim([25,1760])
bias = 0.02#绘制误差图时的每条基准线的偏差

cd_dict = {0:2800,1:2100}#总计的测试电流密度
star_dict = {0:314,1:627,2:938,3:1231,4:1591}#阶跃周期开始
end_dict = {0:541,1:845,2:1149,3:1446,4:1705}#周期末期
Step_dict = {0:700,1:1400,2:1400,3:1400,4:700}#阶跃变化幅度
Step_Direct_dict = {0:'up',1:'down',2:'up',3:'down',4:'up'}#阶跃变化方向

Amp = pandas.DataFrame(columns = ['CurrentDensity','Step','StepDirection','error'])
Bias = pandas.DataFrame(columns = ['CurrentDensity','Step','StepDirection','error'])
for i in range(len(step_seq[5:])):
    i += 4
    step_cur = step_seq[i]
    step_next = step_seq[i+1]

    vol_cur = voltage[step_cur:step_next]
    plt.plot(vol_cur,'o')
    vol_nn = polar_res[step_cur:step_next]
    plt.plot(vol_nn)
    # plt.plot(vol_cur - vol_nn + bias*i,'o')
    # plt.plot([0,1770],[bias*i,bias*i], 'r')
    err = np.array(vol_cur-vol_nn)
    # err_dict[int(round(float(current_density[step_seq[i]]/50))*50)] = err[0:51]#这里是分析极化曲线时候的操作，现在肯定是不合适了

    record_length = 50
    for k in [0,1,2,3,4]:
        for j in range(record_length):
            start = star_dict[k] + j
            end = end_dict[k] + j
            cur = {}
            cur['CurrentDensity'] = cd_dict[i-4]
            cur['Step'] = Step_dict[k]
            cur['StepDirection'] = Step_Direct_dict[k]
            cur['error'] = float(err[start])
            Amp = Amp.append(cur,ignore_index=True)
            cur['error'] = float(err[end])
            Bias = Bias.append(cur,ignore_index=True)

# Amp.to_csv('2s/DynamicResponse/amplitude_1129_second_two.csv')
# Bias.to_csv('2s/DynamicResponse/bias_1129_second_two.csv')
# print(Amp)
# print(Bias)

# df_amp = pandas.read_csv('2s/DynamicResponse/amplitude_1129_second_two.csv')
# df_bias = pandas.read_csv('2s/DynamicResponse/bias_1129_second_two.csv')
#
# plt.figure(figsize=(15, 8))
#
# ec = df_amp['error_corrected']
# ec = abs(ec)
# df_amp['error_corrected'] = ec
# # df_amp.to_csv('2s/DynamicResponse/amplitude_1129_second_two.csv')
# sns.violinplot(x = 'CurrentDensity', y = 'error_corrected', data = df_amp, hue = 'Step')
# # sns.violinplot(x = 'CurrentDensity', y = 'error', data = df_bias, hue = 'StepDirection')
# plt.title('Step error of 1129 second')
# plt.subplots_adjust(left=0.073, bottom=0.062, right=0.95, top=0.925)

'''合并一天中的两部分，并进行分析'''
# df1 = pandas.read_csv('2s/DynamicResponse/amplitude_1129_first_three.csv')
# df2 = pandas.read_csv('2s/DynamicResponse/amplitude_1129_second_two.csv')
#
# df_all = df1.append(df2)
#
# df_all.to_csv('2s/DynamicResponse/amplitude_1129_all.csv')

df = pandas.read_csv('2s/DynamicResponse/amplitude_1129_all.csv')

ec = df['error_corrected']
ec = abs(ec)
df['error_corrected'] = ec

plt.figure(figsize=(15, 8))
plt.title('1129-all')
plt.subplots_adjust(left=0.073, bottom=0.062, right=0.95, top=0.925)
sns.violinplot(x = 'CurrentDensity', y = 'error_corrected', data = df, hue = 'Step')











"""别的日期"""




plt.show()