'''
这里主要是想尝试进行一下重新对极化曲线进行拟合，因为之前的方程只考虑了出口温度，这里需要通过合理的函数，把入口温度和出口温度都考虑进去
'''
import numpy
import numpy as np
import pandas as np
import pandas
import matplotlib.pyplot as plt
import pandas as pd

import time
import os


def Vres(Temp):
    '''这里就是计算热中性电压'''
    import numpy as np
    T_ref = 25
    F = 96485
    n = 2
    R = 8.3145
    CH2O = 75   #参考点状态下的水热容(单位：J/(K*mol))
    CH2  = 29
    CO2 = 29
    S0_H2 = 131
    S0_H20 = 70
    S0_O2 = 205
    DHH2O =-2.86*10**5 + CH2O * (Temp - T_ref)    #参考点状态下的焓变(单位：J/mol)
    DHH2 = 0  + CH2 * (Temp- T_ref) #参考点状态下的焓变(单位：J/mol)
    DHO2 = 0  + CO2 * (Temp - T_ref)  #参考点状态下的焓变(单位：J/mol)
    DH = DHH2 + DHO2/2 - DHH2O

    SH2 = CH2 * np.math.log((Temp + 273.15) / (T_ref + 273.15),10) - R * np.math.log(10,10) +S0_H2
    SO2 = CO2 * np.math.log((Temp + 273.15) / (T_ref + 273.15),10) - R * np.math.log(10,10) +S0_O2
    SH20 = CH2O * np.math.log((Temp + 273.15) / (T_ref + 273.15),10) + S0_H20
    DS = SH2 + 0.5*SO2 - SH20
    DG = DH - (Temp+ 273.15) * DS

    return DG / (n*F)

def polar_lihao(Temp,current):
       import math
       r1 = 0.0001362
       r2 = -1.316e-06
       s1 = 0.06494
       s2 = 0.0013154
       s3 = -4.296e-06
       t1 = 0.1645
       t2 = -18.96
       t3 = 672.5
       j = current / 0.425
       U = Vres(Temp) + (r1 + r2 * Temp) * j + (s1 + s2 * Temp + s3 * Temp ** 2) * math.log(((t1 + t2 / Temp + t3 / Temp ** 2) * j + 1))
       return U
def retrive_polar_v_timesequence(T_out_seq,current,n_cell = 34):
       re_V = []
       for i in range(len(T_out_seq)):
              re_V.append(polar_lihao(T_out_seq[i],current[i])*n_cell)
       return re_V


def compare_polar_and_original(V_res,voltage):

    plt.figure(figsize=(15, 8))
    ax1 = plt.gca()
    ax1.plot(voltage)
    ax1.plot(V_res)
    ax1.legend(['original', 'fitted'])
    ax1.set_ylabel('voltage')
    ax1.set_xlabel('Time')

    ax2 = ax1.twinx()
    ax2.set_ylabel('voltage error')
    error = V_res - voltage
    ax2.plot([0] * len(error), color='r',alpha = 0.7)
    ax2.scatter(range(len(error)),error , alpha=0.3)
    plt.title('recovered voltage')
    plt.legend(['error'],loc = 9)
    plt.subplots_adjust(left=0.057, bottom=0.062, right=0.95, top=0.95)
    plt.show()

t0 = time.time()
SourceFile = '20s/Polar Fitting/Polar Tests.csv'

df = pandas.read_csv(SourceFile)
OfficialColumns = ['T_in', 'T_out', 'Current', 'Voltage']
voltage = np.array(df['Voltage'])
current = np.array(df['Current'])
T_in = np.array(df['T_in'])
T_out = np.array(df['T_out'])

current_density = current/0.425
voltage /= 34

Ures = []
for i in range(len(T_out)):
    Ures.append(Vres((T_out[i] + T_in[i])/2))

import tensorflow as tf


'''ar1=tf.Variable(initial_value=2.0066625e-05)
ar2=tf.Variable(initial_value=4.1251087e-07)
ar3=tf.Variable(initial_value=4.1251087e-10)
as1=tf.Variable(initial_value=0.06486329)
as2=tf.Variable(initial_value=0.0013190222)
as3=tf.Variable(initial_value=-1.9178371e-05)
at1=tf.Variable(initial_value=0.16224697)
at2=tf.Variable(initial_value=-18.962816)
at3=tf.Variable(initial_value=668.4519)
br1=tf.Variable(initial_value=-9.6286585e-06)
br2=tf.Variable(initial_value=6.4695615e-07)
bs1=tf.Variable(initial_value=0.034834314)
bs2=tf.Variable(initial_value=0.0005985674)
bs3=tf.Variable(initial_value=-1.3294286e-06)
bt1=tf.Variable(initial_value=0.08177479)
bt2=tf.Variable(initial_value=-9.954222)
bt3=tf.Variable(initial_value=351.5163)'''

ar1=tf.Variable(initial_value=2.016921e-05)
ar2=tf.Variable(initial_value=5.7740976e-07)
ar3=tf.Variable(initial_value=5.070187e-10)
as1=tf.Variable(initial_value=0.06486329)
as2=tf.Variable(initial_value=0.0013190222)
as3=tf.Variable(initial_value=-1.9174786e-05)
at1=tf.Variable(initial_value=0.16224697)
at2=tf.Variable(initial_value=-18.962816)
at3=tf.Variable(initial_value=668.4519)
br1=tf.Variable(initial_value=-9.531767e-06)
br2=tf.Variable(initial_value=3.538604e-07)
bs1=tf.Variable(initial_value=0.034834314)
bs2=tf.Variable(initial_value=0.0005985674)
bs3=tf.Variable(initial_value=-1.0110334e-06)
bt1=tf.Variable(initial_value=0.08177479)
bt2=tf.Variable(initial_value=-9.954222)
bt3=tf.Variable(initial_value=351.5163)



variables = [ar2,ar3,  as2, as3,  at2, at3,  br2,  bs2, bs3,  bt2, bt3]

optimizer = tf.keras.optimizers.SGD(learning_rate = 1E-15)
#optimizer = tf.keras.optimizers.Adadelta(learning_rate=0.00001,rho=0.95,epsilon=1e-07,name='Adadelta')
num_epoch = 0
loss_seq = []
batch_size = 50
factor = 1
train = 1
if train == 1:
    for i in range(num_epoch):
        start = 0
        L_accumulate = 0
        while start<len(voltage):
            with tf.GradientTape() as tape:
                V_pred = Ures + (ar1 + ar2 * T_out + ar3 * T_out**2) * current_density
                V_pred += (as1 + as2 * T_out + as3 * T_out ** 2) * tf.math.log((at1 + at2 / T_out + at3 / T_out ** 2) * current_density + 1)
                V_pred += (br1 + br2 * T_in) * current_density
                V_pred += (bs1 + bs2 * T_in + bs3 * T_in ** 2) * tf.math.log(
                    (bt1 + bt2 / T_in + bt3 / T_in ** 2) * current_density + 1)
                V_pred_cur = V_pred[start:start + batch_size]
                V_cur = voltage[start:start + batch_size]
                L_cur = tf.reduce_sum(tf.abs(V_pred_cur - V_cur) )/batch_size/ factor
            grads = tape.gradient(L_cur, variables)
            optimizer.apply_gradients(grads_and_vars=zip(grads, variables))
            start += batch_size
            L_accumulate += L_cur
        loss_seq.append(L_accumulate * factor)
        print(str(i) + '\t' + 'loss = ' + str(L_accumulate.numpy() ))



    variables = [ar1, ar2,ar3, as1, as2, as3, at1, at2, at3, br1, br2, bs1, bs2, bs3, bt1, bt2, bt3]
    key_variables = ['ar1', 'ar2','ar3', 'as1', 'as2', 'as3', 'at1', 'at2', 'at3', 'br1', 'br2', 'bs1', 'bs2', 'bs3', 'bt1',
                     'bt2', 'bt3']
    for v in range(len(variables)):
        #print(key_variables[v] +'=tf.Variable(initial_value='  + str(variables[v].numpy()) + ')')
        print(key_variables[v] + ' \t' + str(variables[v].numpy()) )

pred = 1
if pred == 1:
    V_res = Ures + (ar1 + ar2 * T_out+ ar3 * T_out**2) * current_density
    V_res += (as1 + as2 * T_out + as3 * T_out ** 2) * tf.math.log(
        (at1 + at2 / T_out + at3 / T_out ** 2) * current_density + 1)
    V_res += (br1 + br2 * T_in) * current_density
    V_res += (bs1 + bs2 * T_in + bs3 * T_in ** 2) * tf.math.log(
        (bt1 + bt2 / T_in + bt3 / T_in ** 2) * current_density + 1)
    print('Epoches = ',num_epoch,'time = ',time.time() - t0)
    compare_polar_and_original(V_res,voltage)