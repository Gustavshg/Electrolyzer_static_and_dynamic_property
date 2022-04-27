"""对比一下之前使用神经网络模型与极化曲线公式的结果"""
import time
import numpy as np
import pandas
from matplotlib import pyplot as plt
import Polar_fitting_collection as polar_collection
import Polar_data_loader

model_nn = polar_collection.polar_nn()


test_data = Polar_data_loader.Original_Data_Loader(SourceFile='2s/Original/TJ-20211129.csv')

T_out, T_in,current_density,LyeFlow = test_data.get_polar_data()
voltage = test_data.get_voltage()
inputs,outpust = test_data.get_all_data()

# train = 0
# if train == 1:
#     import tensorflow as tf
#     optimizer = tf.keras.optimizers.Adam(learning_rate = 1E-3)
#     inputs,outputs = test_data.get_batch()
#     with tf.GradientTape() as tape:
#         v_pred = model_nn(T_out,current_density)
#     grads = tape.gradient(v_pred,model_nn.variables)
#     print(grads)

comparison = 1
if comparison ==1:
    import tensorflow as tf
    polar_lihao =  polar_collection.polar_lihao(T_out,current_density)
    polar_wtt =  polar_collection.polar_wtt(T_out,current_density)
    polar_shg = polar_collection.polar_shg(T_out,T_in,current_density,LyeFlow)
    polar_nn_shg = model_nn.predict(inputs)

    error_LH = polar_lihao - voltage
    error_WTT = polar_wtt - voltage
    error_SHG = polar_shg - voltage
    error_NN = polar_nn_shg - voltage

    loss_LH = tf.reduce_sum(tf.square(error_LH)).numpy()
    loss_WTT = tf.reduce_sum(tf.square(error_WTT)).numpy()
    loss_SHG = tf.reduce_sum(tf.square(error_SHG)).numpy()
    loss_NN = tf.reduce_sum(tf.square(error_NN)).numpy()
    """画四条线的直接对比图"""
    plt.figure(figsize=(15, 8))
    ax1 = plt.gca()

    ax1.plot(polar_lihao,color ='gold')
    ax1.plot(polar_wtt,color ='lightgreen')
    ax1.plot(polar_shg,color ='turquoise')
    ax1.plot(polar_nn_shg,color ='violet')
    ax1.plot(voltage,color ='red',alpha = 0.7)
    ax1.legend([ 'LH', 'WTT', 'SHG', 'NN','original'], loc=0)
    ax1.set_ylabel('voltage')
    ax1.set_ylim([0, 2.2])
    ax1.set_xlabel('Time')
    ax2 = ax1.twinx()
    ax2.set_ylabel('voltage error')
    alpha = 0.2
    ax2.plot([0] * len(voltage), color='r',alpha = 0.7)
    ax2.scatter(range(len(voltage)), error_LH,color ='gold', alpha=alpha)
    print('loss of LiHao = ', loss_LH)
    ax2.scatter(range(len(voltage)), error_WTT,color ='lightgreen', alpha=alpha)
    print('loss of WangTiantian = ',loss_WTT)
    ax2.scatter(range(len(error_SHG)), error_SHG,color ='turquoise', alpha=alpha)
    print('loss of SHG = ', loss_SHG)
    ax2.scatter(range(len(error_NN)), error_NN,color ='violet', alpha=alpha)
    print('loss of Neural Network = ', loss_NN)
    plt.title('comparison between polar curves')
    plt.legend(['baseline','LH', 'WTT', 'SHG', 'NN'], loc=9,framealpha=0.5)
    plt.subplots_adjust(left=0.057, bottom=0.062, right=0.95, top=0.95)
    """画总体的loss对比"""
    plt.figure(figsize=(8, 4))
    x_axis = ['LH', 'WTT', 'SHG', 'NN']
    y_axis = [loss_LH,loss_WTT,loss_SHG,loss_NN]
    plt.bar( x = x_axis,height = y_axis ,width = 0.5)
    plt.subplots_adjust(left=0.057, bottom=0.062, right=0.95, top=0.95)
    plt.title('loss comparison between polar curves')
    for x,y in zip(x_axis,y_axis):
        plt.text(x,y,int(y*1000)/1000.,ha = 'center', va = 'bottom')
    """画误差的histogram"""
    plt.figure(figsize=(8, 4))
    import seaborn as sns

    sns.set_style('darkgrid')
    sns.distplot(error_LH)
    sns.distplot(error_WTT)
    sns.distplot(error_SHG)
    sns.distplot(error_NN)
    plt.legend(['LH', 'WTT', 'SHG', 'NN'])
    plt.subplots_adjust(left=0.073, bottom=0.062, right=0.95, top=0.925)
    plt.title('loss comparison between polar curves')
    plt.xlim([-0.4,0.4])
    plt.show()