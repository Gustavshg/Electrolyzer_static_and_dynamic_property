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

'''including more data'''
'''SelectedFiles = ["20s/Original/TJ-20211202.csv"]
OriginalColumns = [ '时间', '电解电压', '电解电流', '产氢量', '产氢累计量', '碱液流量', '碱温',
       '系统压力  ', '氧槽温', '氢槽温', '氧侧液位', '氢侧液位', '氧中氢', '氢中氧', '脱氧上温', '脱氧下温',
       'B塔上温', 'B塔下温', 'C塔上温', 'C塔下温', 'A塔上温', 'A塔下温', '露点', '微氧量', '出罐压力',
       '进罐温度', '进罐压力']
df = pandas.read_csv(SelectedFiles[0])
df['T_out'] = ( df['氧槽温'] + df['氢槽温'])/2
df['T_in'] = df['碱温']
df['Current'] = df['电解电流']
df['Voltage'] = df['电解电压']

T_in = np.array(df['T_in'])
T_out = np.array(df['T_out'])
current = np.array(df['Current'])
voltage = np.array(df['Voltage'])
data = np.array([T_in,T_out])
newdf = pandas.DataFrame()
newdf['T_in'] = T_in
newdf['T_out'] = T_out
newdf['current'] = current
newdf['voltage'] = voltage
tofiledf = pandas.DataFrame()
plt.plot(current)
tofiledf = newdf[2200:2300]
#tofiledf = tofiledf.append(newdf[4000:4100])
#tofiledf = tofiledf.append(newdf[400:450])
#tofiledf = tofiledf.append(newdf[510:560])
#tofiledf = tofiledf.append(newdf[620:670])
#tofiledf = tofiledf.append(newdf[715:755])
#tofiledf = tofiledf.append(newdf[800:840])
#tofiledf = tofiledf.append(newdf[920:970])
#tofiledf = tofiledf.append(newdf[1100:1200])
#tofiledf = tofiledf.append(newdf[1250:1300])
#tofiledf.to_csv('20s/1130 Polar data.csv')
print(tofiledf)
plt.show()'''

'''summerizing all the polar data'''

'''inputDateFolder = "20s/Polar Fitting"
inputDateFiles =  os.listdir(inputDateFolder)
OfficialColumns = ['','T_in', 'T_out', 'Current', 'Voltage']
tofiledf = pandas.DataFrame()
for file in inputDateFiles:
    if not file == 'Polar Tests.csv' and  not file == 'All dates data.csv':
        df = pandas.read_csv(os.path.join(inputDateFolder,file))
        df.columns = OfficialColumns
        tofiledf = tofiledf.append(df)
tofiledf.index = range(len(tofiledf))
plt.plot(tofiledf['Current'])
plt.show()
tofiledf.to_csv('20s/Polar Fitting/All dates data.csv')'''

'''showing data'''
'''SourceFile = '20s/Polar Fitting/Polar Tests.csv'

df = pandas.read_csv(SourceFile)
df.index = range(len(df))
print(len(df))
OfficialColumns = ['T_in', 'T_out', 'Current', 'Voltage']
plt.plot(df['Current'])
plt.xlabel('Time')
plt.ylabel('Current')
plt.title('All dates data')
plt.show()'''

'''inputting data'''
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

'''李昊的极化曲线参数'''
'''ar1 = tf.Variable(initial_value=0.0001362)
ar2 = tf.Variable(initial_value=-1.316e-06)
as1 = tf.Variable(initial_value=0.06494)
as2 = tf.Variable(initial_value=0.001354)
as3 = tf.Variable(initial_value=-4.296e-06)
at1 = tf.Variable(initial_value=0.1645)
at2 = tf.Variable(initial_value=-18.96)
at3 = tf.Variable(initial_value=672.5)'''

'''ar1=tf.Variable(initial_value=6.42075e-05)
ar2=tf.Variable(initial_value=1.238423e-07)
as1=tf.Variable(initial_value=0.06494)
as2=tf.Variable(initial_value=0.0013538734)
as3=tf.Variable(initial_value=-1.831564e-05)
at1=tf.Variable(initial_value=0.1645)
at2=tf.Variable(initial_value=-18.96)
at3=tf.Variable(initial_value=672.5)
br1=tf.Variable(initial_value=3.4612745e-05)
br2=tf.Variable(initial_value=-4.698827e-07)
bs1=tf.Variable(initial_value=0.03494)
bs2=tf.Variable(initial_value=0.0006540465)
bs3=tf.Variable(initial_value=-6.06711e-06)
bt1=tf.Variable(initial_value=0.0845)
bt2=tf.Variable(initial_value=-9.96)
bt3=tf.Variable(initial_value=350.5)'''

'''ar1=tf.Variable(initial_value=6.1582476e-05)#SDG 10000次结果
ar2=tf.Variable(initial_value=5.7367447e-07)
as1=tf.Variable(initial_value=0.06494)
as2=tf.Variable(initial_value=0.0013538734)
as3=tf.Variable(initial_value=-1.5524221e-05)
at1=tf.Variable(initial_value=0.1645)
at2=tf.Variable(initial_value=-18.96)
at3=tf.Variable(initial_value=672.5)
br1=tf.Variable(initial_value=3.188635e-05)
br2=tf.Variable(initial_value=-1.1717486e-06)
bs1=tf.Variable(initial_value=0.03494)
bs2=tf.Variable(initial_value=0.0006534644)
bs3=tf.Variable(initial_value=-7.929595e-06)
bt1=tf.Variable(initial_value=0.0845)
bt2=tf.Variable(initial_value=-9.96)
bt3=tf.Variable(initial_value=350.5)'''

'''ar1=tf.Variable(initial_value=1.9738643e-05)#Adadelta 10000次结果
ar2=tf.Variable(initial_value=-3.101854e-07)
as1=tf.Variable(initial_value=0.06486329)
as2=tf.Variable(initial_value=0.0013200146)
as3=tf.Variable(initial_value=-1.6814276e-05)
at1=tf.Variable(initial_value=0.16059265)
at2=tf.Variable(initial_value=-18.965357)
at3=tf.Variable(initial_value=672.5)
br1=tf.Variable(initial_value=-9.956674e-06)
br2=tf.Variable(initial_value=1.380596e-06)
bs1=tf.Variable(initial_value=0.034834314)
bs2=tf.Variable(initial_value=0.00059870764)
bs3=tf.Variable(initial_value=-3.4011491e-06)
bt1=tf.Variable(initial_value=0.078851596)
bt2=tf.Variable(initial_value=-9.963849)
bt3=tf.Variable(initial_value=350.5)'''

'''ar1=tf.Variable(initial_value=-7.829887e-05)
ar2=tf.Variable(initial_value=1.7041887e-06)
as1=tf.Variable(initial_value=0.064409696)
as2=tf.Variable(initial_value=0.0011395052)
as3=tf.Variable(initial_value=-1.7382554e-05)
at1=tf.Variable(initial_value=1.4168991)
at2=tf.Variable(initial_value=1.9623994)
br1=tf.Variable(initial_value=-0.00010799364)
br2=tf.Variable(initial_value=3.3217912e-06)
bs1=tf.Variable(initial_value=0.03438993)
bs2=tf.Variable(initial_value=0.00038729463)
bs3=tf.Variable(initial_value=-1.9448617e-05)
bt1=tf.Variable(initial_value=1.3516635)
bt2=tf.Variable(initial_value=1.9526393)'''

ar1=tf.Variable(initial_value=-0.000102254955)
ar2=tf.Variable(initial_value=-2.464152e-06)
as1=tf.Variable(initial_value=0.064409696)
as2=tf.Variable(initial_value=0.0011307291)
as3=tf.Variable(initial_value=-8.88342e-06)
at1=tf.Variable(initial_value=0.5576032)
at2=tf.Variable(initial_value=-4.755861)
br1=tf.Variable(initial_value=-0.00013154959)
br2=tf.Variable(initial_value=8.882304e-06)
bs1=tf.Variable(initial_value=0.03438993)
bs2=tf.Variable(initial_value=0.00037977129)
bs3=tf.Variable(initial_value=-2.9856506e-05)
bt1=tf.Variable(initial_value=0.27493647)
bt2=tf.Variable(initial_value=-3.3638756)




'''variables = [ar1,ar2,as1,as2,as3,at1,at2,br1,br2,bs1,bs2,bs3,bt1,bt2]
key_variables = ['ar1','ar2','as1','as2','as3','at1','at2','br1','br2','bs1','bs2','bs3','bt1','bt2']'''


variables = [at2,bt2]
key_variables = ['at2','bt2']

#optimizer = tf.keras.optimizers.SGD(learning_rate = 1E5)
optimizer = tf.keras.optimizers.Adadelta(learning_rate=0.001,rho=0.95,epsilon=1e-07,name='Adadelta')
num_epoch = 1500
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
                V_pred = Ures + (ar1 + ar2 * T_out) * current_density
                V_pred += (as1 + as2 * T_out + as3 * T_out ** 2) * tf.math.log((at1 + at2 / T_out ) * current_density + 1)
                V_pred += (br1 + br2 * T_in) * current_density
                V_pred += (bs1 + bs2 * T_in + bs3 * T_in ** 2) * tf.math.log(
                    (bt1 + bt2 / T_in ) * current_density + 1)
                V_pred_cur = V_pred[start:start + batch_size]
                V_cur = voltage[start:start + batch_size]
                L_cur = tf.reduce_sum(tf.square(V_pred_cur - V_cur) )/batch_size/ factor
            grads = tape.gradient(L_cur, variables)
            optimizer.apply_gradients(grads_and_vars=zip(grads, variables))
            start += batch_size
            L_accumulate += L_cur
        loss_seq.append(L_accumulate * factor)
        print(str(i) + '\t' + 'loss = ' + str(L_accumulate.numpy() ))

    for g in range(len(variables)):
        print(key_variables[g] +'\t'+ 'gradient is ' + str(grads[g].numpy()) )

    variables = [ar1, ar2, as1, as2, as3, at1, at2, br1, br2, bs1, bs2, bs3, bt1, bt2]
    key_variables = ['ar1', 'ar2', 'as1', 'as2', 'as3', 'at1', 'at2',  'br1', 'br2', 'bs1', 'bs2', 'bs3', 'bt1',
                     'bt2']
    for v in range(len(variables)):
        print(key_variables[v] +'=tf.Variable(initial_value='  + str(variables[v].numpy()) + ')')

    '''with open('20s/Polar Fitting/Polar Variables 0326.txt', 'w') as f:
        localtime = time.asctime(time.localtime(time.time()))
        f.write(str(localtime) + '\n')
        for v in range(len(variables)):
            f.write(key_variables[v] + '=tf.Variable(initial_value=' + str(variables[v].numpy()) + ')' + '\n')'''
pred = 1
if pred == 1:
    V_res = Ures
    V_res+=  (ar1 + ar2 * T_out) * current_density
    V_res += (as1 + as2 * T_out + as3 * T_out ** 2) * tf.math.log((at1 + at2 / T_out ) * current_density + 1)
    V_res += (br1 + br2 * T_in) * current_density
    V_res += (bs1 + bs2 * T_in + bs3 * T_in ** 2) * tf.math.log((bt1 + bt2 / T_in ) * current_density + 1)
    print(time.time() - t0)
    compare_polar_and_original(V_res,voltage)