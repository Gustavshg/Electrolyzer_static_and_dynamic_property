'''
这里主要是想尝试进行一下重新对极化曲线进行拟合，因为之前的方程只考虑了出口温度，这里需要通过合理的函数，把入口温度和出口温度都考虑进去
在0403的版本里面，我们发现了碱液流量在两天数据中的不同，所以我们需要把碱液流量通过合理的方法考虑进去，这就需要我们重新全部重做一下之前的极化曲线的数据集
'''
import numpy
import numpy as np
import pandas as np
import pandas
import matplotlib.pyplot as plt
import random
import time
import os


def varname(var,all_var=locals()):
    return [varname for varname in all_var if all_var[varname] is var][0]


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
    plt.title('recovered voltage-t')
    plt.legend(['error'],loc = 9)
    plt.subplots_adjust(left=0.057, bottom=0.062, right=0.95, top=0.95)
    plt.show()

'''开始拟合部分'''
t0 = time.time()
SourceFile = '20s/Polar Fitting/Polar Test data.csv'

df = pandas.read_csv(SourceFile)

voltage = np.array(df['Voltage'])
current = np.array(df['Current'])
T_in = np.array(df['T_in'])
T_out = np.array(df['T_out'])
LyeFlow = np.array(df['LyeFlow'])
voltage /= 34
current_density = current/0.425

voltage_ori = np.array(df['Voltage'])
current_ori = np.array(df['Current'])
T_in_ori = np.array(df['T_in'])
T_out_ori = np.array(df['T_out'])
LyeFlow_ori = np.array(df['LyeFlow'])
current_density_ori = current_ori/0.425
voltage_ori /=34

Ures = []
for i in range(len(T_out)):
    Ures.append(Vres((T_out[i] + T_in[i])/2))



import tensorflow as tf

'''原始方程参数'''
r1=tf.Variable(initial_value=2.020922e-05)
r2=tf.Variable(initial_value=9.82668e-07)
r3=tf.Variable(initial_value=-5.4770095e-09)
s1=tf.Variable(initial_value=0.06486329)
s2=tf.Variable(initial_value=0.0013241252)
s3=tf.Variable(initial_value=-1.5643182e-05)
t1=tf.Variable(initial_value=0.17286256)
t2=tf.Variable(initial_value=-19.080109)
t3=tf.Variable(initial_value=668.4519)


variables =  [t1,t2,t3]

'''给入口温度乘上碱液流量直接进行拟合'''
T_ave  = (T_out + T_in)/2
T_ave_ori = (T_out + T_in)/2


'''优化器设置'''
optimizer = tf.keras.optimizers.SGD(learning_rate = 1E-2)
#optimizer = tf.keras.optimizers.Adadelta(learning_rate=0.0001,rho=0.95,epsilon=1e-07,name='Adadelta')

'''训练设置'''
num_epoch = 1000
loss_seq = []
batch_size = 50
factor = 1
train = 1
if train == 1:
    for i in range(num_epoch):
        start = 0
        L_accumulate = 0
        '''在每次训练中，都对数据进行随机穿梭'''
        seed = random.randint(0,999)
        random.seed(seed)
        random.shuffle(T_out)
        random.shuffle(T_in)
        random.shuffle(voltage)
        random.shuffle(current_density)
        random.shuffle(LyeFlow)
        while start<len(voltage):
            with tf.GradientTape() as tape:
                '''极化曲线方程'''
                V_pred = Ures + current_density * (r1 + r2 * T_ave + r3 * T_ave**2) + (s1 + s2 * T_ave + s3 * T_ave**2) \
                         * tf.math.log( current_density * (t1 + t2 /T_ave + t3 / T_ave**2) + 1)


                '''取批次进行训练'''
                V_pred_cur = V_pred[start:start + batch_size] # 预测值
                V_cur = voltage[start:start + batch_size] #label，目标值
                '''计算当前批次的损失函数结果'''
                L_cur = tf.reduce_sum(tf.square(V_pred_cur - V_cur) )/batch_size/ factor
            '''对损失函数进行求导'''
            grads = tape.gradient(L_cur, variables)
            optimizer.apply_gradients(grads_and_vars=zip(grads, variables))
            start += batch_size
            L_accumulate += L_cur
        loss_seq.append(L_accumulate * factor)
        print(str(i) + '\t' + 'loss = ' + str(L_accumulate.numpy() ))



    all_variables = [r1,r2,r3,s1,s2,s3,t1,t2,t3]

    for v in range(len(all_variables)):
        print(varname(all_variables[v]) +'=tf.Variable(initial_value='  + str(all_variables[v].numpy()) + ')')

T_out = T_out_ori
T_in = T_in_ori
current_density = current_density_ori
voltage = voltage_ori
LyeFlow = LyeFlow_ori

pred = 1
if pred == 1:
    V_pred = Ures + current_density * (r1 + r2 * T_ave + r3 * T_ave ** 2) + (s1 + s2 * T_ave + s3 * T_ave ** 2) \
             * tf.math.log(current_density * (t1 + t2 / T_ave + t3 / T_ave ** 2) + 1)
    print('Epoches = ',num_epoch,'time = ',time.time() - t0)
    compare_polar_and_original(V_pred,voltage_ori)






