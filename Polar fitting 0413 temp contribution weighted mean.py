'''
这里主要是想尝试进行一下重新对极化曲线进行拟合，因为之前的方程只考虑了出口温度，这里需要通过合理的函数，把入口温度和出口温度都考虑进去
在0403的版本里面，我们发现了碱液流量在两天数据中的不同，所以我们需要把碱液流量通过合理的方法考虑进去，这就需要我们重新全部重做一下之前的极化曲线的数据集
'''
import math
import numpy as np
import pandas
import matplotlib.pyplot as plt
import time


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
    plt.title('recovered voltage-Weighted average by cooling ratio SGD rs/t step by step')
    plt.legend(['error'],loc = 9)
    plt.subplots_adjust(left=0.057, bottom=0.062, right=0.95, top=0.95)
    plt.show()

'''开始拟合部分'''
t0 = time.time()
# SourceFile = '20s/Polar Fitting/Polar Test data.csv'
SourceFile = '20s/Polar Fitting/All Polar-like data.csv' #和0413版本的主要区别就在于更换了数据
df = pandas.read_csv(SourceFile)

voltage = np.array(df['Voltage'])
current = np.array(df['Current'])
T_in = np.array(df['T_in'])
T_out = np.array(df['T_out'])
LyeFlow = np.array(df['LyeFlow'])
voltage /= 34
current_density = current/0.425

'''冷却流量比'''
lam = 0.204425/LyeFlow

'''给入口温度乘上碱液流量直接进行拟合'''
import tensorflow as tf
# T_ave  = (T_out + T_in)/2
T_ave = lam * T_out + (1-lam) * T_in
# T_ave = np.sqrt(lam * T_out ** 2 + (1-lam) * T_in ** 2)

Ures = []
for i in range(len(T_out)):
    Ures.append(Vres(T_ave[i]))

'''原始方程参数'''
ar1=tf.Variable(initial_value=7.39445e-05)
ar2=tf.Variable(initial_value=5.759043e-06)
ar3=tf.Variable(initial_value=-5.5199894e-08)
as1=tf.Variable(initial_value=0.06491715)
as2=tf.Variable(initial_value=0.00036700224)
as3=tf.Variable(initial_value=-1.561482e-05)
at1=tf.Variable(initial_value=0.19111753)
at2=tf.Variable(initial_value=-47.666733)
at3=tf.Variable(initial_value=3657.258)
br1=tf.Variable(initial_value=6.4338914e-05)
br2=tf.Variable(initial_value=5.231239e-07)
br3=tf.Variable(initial_value=-1.3664759e-08)
bs1=tf.Variable(initial_value=0.06475956)
bs2=tf.Variable(initial_value=0.0001710047)
bs3=tf.Variable(initial_value=9.757855e-06)
bt1=tf.Variable(initial_value=0.19021238)
bt2=tf.Variable(initial_value=-26.661955)
bt3=tf.Variable(initial_value=1217.563)
# Overall loss =  0.23292498



variables = [ar1, ar2,ar3, as1, as2, as3, at1, at2, at3, br1, br2, br3, bs1, bs2, bs3, bt1, bt2, bt3]
# variables = [ ar3,br3]
# variables = [at2, at3, bt2, bt3]
# variables = [as2, as3, bs2, bs3]
# variables = [ as1,bs1]
# variables = [ at1,bt1]

'''优化器设置'''
optimizer = tf.keras.optimizers.Adam(learning_rate = 1E-10)
# optimizer = tf.keras.optimizers.SGD(learning_rate = 1E-13)
# optimizer = tf.keras.optimizers.Adadelta(learning_rate=0.001,rho=0.95,epsilon=1e-07,name='Adadelta')

'''训练设置'''
num_epoch = 00
loss_seq = []
batch_size = 50
factor = 1
train = 1
if train == 1:
    for i in range(num_epoch):
        start = 0
        L_accumulate = 0
        '''在每次训练中，都对数据进行随机穿梭'''
        while start<len(voltage):
            with tf.GradientTape() as tape:
                '''极化曲线方程'''
                V_out = current_density * (ar1 + ar2 * T_out + ar3 * T_out**2) + (as1 + as2 * T_out + as3 * T_out**2) \
                         * tf.math.log( current_density * (at1 + at2 /T_out + at3 / T_out**2) + 1)
                V_in  = current_density * (br1 + br2 * T_in + br3 * T_in**2) + (bs1 + bs2 * T_in + bs3 * T_in**2) \
                         * tf.math.log( current_density * (bt1 + bt2 /T_in + bt3 / T_in**2) + 1)
                V_pred = Ures + lam * V_out + (1-lam) * V_in
                '''取批次进行训练'''
                V_pred_cur = V_pred[start:start + batch_size] # 预测值
                V_cur = voltage[start:start + batch_size] #label，目标值
                '''计算当前批次的损失函数结果'''
                L_cur = tf.reduce_sum(tf.square(V_pred_cur - V_cur) )/ factor
            '''对损失函数进行求导'''
            grads = tape.gradient(L_cur, variables)
            optimizer.apply_gradients(grads_and_vars=zip(grads, variables))
            start += batch_size
            L_accumulate += L_cur
        if tf.math.is_nan(L_accumulate):
            #不行了就赶紧收手
            break
        loss_seq.append(L_accumulate * factor)
        print(str(i) + '\t' + 'loss = ' + str(L_accumulate.numpy() ))



    all_variables =[ar1, ar2,ar3, as1, as2, as3, at1, at2, at3, br1, br2, br3, bs1, bs2, bs3, bt1, bt2, bt3]
    for v in range(len(all_variables)):
        print(varname(all_variables[v]) + '= ' + str(all_variables[v].numpy()))
        # print(varname(all_variables[v]) +'=tf.Variable(initial_value='  + str(all_variables[v].numpy()) + ')')


'''重新为之前随机穿梭后的数据赋予原始的值'''

pred = 1
if pred == 1:
    V_out_re = current_density * (ar1 + ar2 * T_out + ar3 * T_out ** 2) + (as1 + as2 * T_out + as3 * T_out ** 2) \
            * tf.math.log(current_density * (at1 + at2 / T_out + at3 / T_out ** 2) + 1)
    V_in_re = current_density * (br1 + br2 * T_in + br3 * T_in ** 2) + (bs1 + bs2 * T_in + bs3 * T_in ** 2) \
           * tf.math.log(current_density * (bt1 + bt2 / T_in + bt3 / T_in ** 2) + 1)
    V_recover = Ures + lam * V_out_re + (1 - lam) * V_in_re
    loss_total = tf.reduce_sum(tf.square(V_recover - voltage))
    print('Overall loss = ', loss_total.numpy())
    print('Epoches = ',num_epoch,'time = ',time.time() - t0)
    compare_polar_and_original(V_recover,voltage)