import Polar_fitting_collection as polar_collection
import Polar_data_loader as loader
import matplotlib.pyplot as plt
import numpy as np

dataloader = loader.Original_Data_Loader()
inputs = dataloader.all_data_x
polar_nn = polar_collection.polar_nn()


current_density = np.arange(0,4000) * 1.
T_out = np.ones(len(current_density)) * 80.0
T_in = np.ones(len(current_density)) * 60.0
LyeFlow = np.ones(len(current_density)) * 1.1
polar_curve_nn = polar_nn.polar(T_out,T_in,current_density,LyeFlow)
polar_curve_shg = polar_collection.polar_shg(T_out,T_in,current_density,LyeFlow)
polar_curve_wtt = polar_collection.polar_wtt(T_out,current_density)
polar_curve_lh = polar_collection.polar_lihao(T_out,current_density)

print(polar_curve_lh)

plt.figure(figsize=(15, 8))
plt.style.use('seaborn')
plt.plot(polar_curve_nn)
plt.plot(polar_curve_shg)
plt.plot(polar_curve_wtt)
plt.plot(polar_curve_lh)
plt.legend(['nn','shg','wtt','lh'])
plt.subplots_adjust(left=0.057, bottom=0.062, right=0.95, top=0.95)
plt.title('comparison between polar curves')

plt.show()