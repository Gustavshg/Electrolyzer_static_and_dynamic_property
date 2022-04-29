import Polar_data_loader
import Polar_fitting_collection as polar_collection
import Polar_data_loader as loader
import matplotlib.pyplot as plt
import numpy as np
import time

dataloader = loader.Original_Data_Loader()
data_polar = loader.Polar_Data_Loader()
inputs = data_polar.all_data_x
T_out, T_in, current_density, LyeFlow = data_polar.get_polar_data()
voltage = data_polar.get_voltage()
polar_nn = polar_collection.polar_nn()

# polar_curve_nn = polar_nn.predict(inputs)
# polar_curve_shg = polar_collection.polar_shg(T_out,T_in,current_density,LyeFlow)
# polar_curve_wtt = polar_collection.polar_wtt(T_out,current_density)
# polar_curve_lh = polar_collection.polar_lihao(T_out,current_density)
# plt.style.use('seaborn')
# fig, ax = plt.subplots(2, 2)
# ax[0,0].scatter(polar_curve_nn,voltage)
# ax[0,0].plot([1.7,2.],[1.7,2.],'r')
# ax[0,1].scatter(polar_curve_shg,voltage)
# ax[0,1].plot([1.7,2.],[1.7,2.],'r')
# ax[1,0].scatter(polar_curve_wtt,voltage)
# ax[1,0].plot([1.7,2.],[1.7,2.],'r')
# ax[1,1].scatter(polar_curve_lh,voltage)
# ax[1,1].plot([1.7,2.],[1.7,2.],'r')
# ax[0, 0].set_title('Neural network')
# ax[0,1].set_title('Shg')
# ax[1, 0].set_title('wtt')
# ax[1, 1].set_title('lh')
# plt.subplots_adjust(left=0.057, bottom=0.062, right=0.95, top=0.95)







plt.style.use('seaborn')
for t_out in [70.,75.,80.,85.]:
    current_density = np.arange(0,4000) * 1.
    T_out = np.ones(len(current_density)) * t_out
    T_in = np.ones(len(current_density)) * 60.0
    LyeFlow = np.ones(len(current_density)) * 1.1
    polar_curve_nn = polar_nn.polar(T_out=T_out, T_in=T_in, current_density=current_density, LyeFlow=LyeFlow)
    plt.plot(polar_curve_nn)
plt.legend([70.,75.,80.,85.])
plt.subplots_adjust(left=0.057, bottom=0.062, right=0.95, top=0.95)
plt.title('polar curves at different temperature')
polar_data = Polar_data_loader.Polar_Data_Loader()
T_out,T_in,current_density,LyeFlow = polar_data.get_polar_data()
voltage = polar_data.get_voltage()
# for i in range(len(T_out)):
#     plt.scatter(current_density[i],voltage[i])
plt.show()


# current_density = np.arange(0,4000) * 1.
# T_out = np.ones(len(current_density)) * 80.0
# T_in = np.ones(len(current_density)) * 60.0
# LyeFlow = np.ones(len(current_density)) * 1.1
# t0 = time.time()
# polar_curve_nn = polar_nn.polar(T_out=T_out,T_in = T_in,current_density=current_density,LyeFlow=LyeFlow)
# t1 = time.time()
# polar_curve_shg = polar_collection.polar_shg(T_out,T_in,current_density,LyeFlow)
# t2 = time.time()
# polar_curve_wtt = polar_collection.polar_wtt(T_out,current_density)
# t3 = time.time()
# polar_curve_lh = polar_collection.polar_lihao(T_out,current_density)
# t4 = time.time()
# print('time of nn is', t1-t0)
# print('time of shg is', t2-t1)
# print('time of wtt is', t3-t2)
# print('time of lh is',t4-t3)

# plt.figure(figsize=(15, 8))
# plt.style.use('seaborn')
# plt.plot(polar_curve_nn)
# plt.plot(polar_curve_shg)
# plt.plot(polar_curve_wtt)
# plt.plot(polar_curve_lh)
# plt.legend(['nn','shg','wtt','lh'])
# plt.subplots_adjust(left=0.057, bottom=0.062, right=0.95, top=0.95)
# plt.title('comparison between polar curves')


# polar_data = Polar_data_loader.Polar_Data_Loader()
# T_out,T_in,current_density,LyeFlow = polar_data.get_polar_data()
# voltage = polar_data.get_voltage()
# for i in range(len(T_out)):
#     plt.scatter(current_density[i],voltage[i])
# print(polar_nn.polar(85.,60.,0.,1.3))

plt.show()