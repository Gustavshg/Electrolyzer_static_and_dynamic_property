'''
这里主要是想尝试进行一下重新对极化曲线进行拟合，因为之前的方程只考虑了出口温度，这里需要通过合理的函数，把入口温度和出口温度都考虑进去
'''
import numpy
import numpy as np
import pandas as np
import pandas
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import time


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

'''inputting data'''
t0 = time.time()
SourceFile = '20s/Polar Fitting/1125 Polar data.csv'
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


'''stochastic gradient descent'''


'''ar1 = tf.Variable(initial_value=0.0000681)
ar2 = tf.Variable(initial_value=-0.616e-06)
as1 = tf.Variable(initial_value=0.06494)
as2 = tf.Variable(initial_value= 0.001354)
as3 = tf.Variable(initial_value=  -4.296e-06)
at1 = tf.Variable(initial_value=0.1645)
at2 = tf.Variable(initial_value=-10.96)
at3 = tf.Variable(initial_value=672.5)
br1 = tf.Variable(initial_value=0.0000681)
br2 = tf.Variable(initial_value=-0.616e-06)
bs1 = tf.Variable(initial_value=0.06494)
bs2 = tf.Variable(initial_value= 0.001354)
bs3 = tf.Variable(initial_value=  -4.296e-06)
bt1 = tf.Variable(initial_value=0.1645)
bt2 = tf.Variable(initial_value=-10.96)
bt3 = tf.Variable(initial_value=672.5)'''

ar1=tf.Variable(initial_value=6.42075e-05)
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
bt3=tf.Variable(initial_value=350.5)







#variables = [ar1,ar2,as1,as2,as3,at1,at2,at3,br1,br2,bs1,bs2,bs3,bt1,bt2,bt3]
#key_variables = ['ar1','ar2','as1','as2','as3','at1','at2','at3','br1','br2','bs1','bs2','bs3','bt1','bt2','bt3']
variables = [ar1,ar2,br1,br2]
key_variables = ['ar1','ar2','br1','br2']

optimizer = tf.keras.optimizers.SGD(learning_rate = 1E-10)
num_epoch = 1
loss_seq = []
batch_size = 50
factor = 1000
train = 1
if train == 1:
    for i in range(num_epoch):
        start = 0
        L_accumulate = 0
        while start<len(voltage):
            with tf.GradientTape() as tape:
                V_pred = Ures + (ar1 + ar2 * T_out) * current_density
                V_pred += (as1 + as2 * T_out +as3* T_out**2) * tf.math.log((at1 + at2 / T_out + at3 / T_out**2) * current_density + 1)
                V_pred +=(br1 + br2 * T_in) * current_density
                V_pred += (bs1 + bs2 * T_in + bs3* T_in **2 ) * tf.math.log((bt1 + bt2 / T_in + bt3 / T_in**2) * current_density + 1)
                V_pred_cur = V_pred[start:start + batch_size]
                V_cur = voltage[start:start + batch_size]
                L = tf.reduce_sum(tf.square(  V_pred_cur - V_cur )/factor)
            grads = tape.gradient(L,variables)
            optimizer.apply_gradients(grads_and_vars=zip(grads,variables))
            start+= batch_size
            L_accumulate+=L
        loss_seq.append(L_accumulate*factor)

        print(str(i) +'\t'+ 'loss = ' + str(L_accumulate.numpy()*factor))

    '''with open('Polar Variables 0326.txt', 'w') as f:
        localtime = time.asctime(time.localtime(time.time()))
        f.write(str(localtime) + '\n')
        for v in range(len(variables)):
            f.write(key_variables[v] +'=tf.Variable(initial_value='  + str(variables[v].numpy()) + ')' + '\n')'''

    for v in range(len(variables)):
        print(key_variables[v] +'=tf.Variable(initial_value='  + str(variables[v].numpy()) + ')')

    print(time.time() - t0)
'''    plt.axes(yscale='log')
    plt.plot(loss_seq, '-')
    plt.plot([100] * num_epoch, 'r')
    plt.plot([10] * num_epoch, 'r')
    plt.plot([1] * num_epoch, 'r')
    plt.plot([0.1] * num_epoch,'r')
    plt.plot([0.01] * num_epoch, 'r')
    #plt.plot([0.001] * num_epoch, 'r')
    #plt.plot([0.0001] * num_epoch, 'r')
    plt.grid()
    plt.title('batch size = 100, loss ,learning rate = 1E-9')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.show()'''

pred = 1

if pred == 1:
    ax1 = plt.gca()
    ax1.plot(voltage)
    V_pred = Ures
    V_pred+=  (ar1 + ar2 * T_out) * current_density
    V_pred += (as1 + as2 * T_out + as3 * T_out ** 2) * tf.math.log(
        (at1 + at2 / T_out + at3 / T_out ** 2) * current_density + 1)
    V_pred += (br1 + br2 * T_in) * current_density
    V_pred += (bs1 + bs2 * T_in + bs3 * T_in ** 2) * tf.math.log(
        (bt1 + bt2 / T_in + bt3 / T_in ** 2) * current_density + 1)
    ax1.plot(V_pred)
    ax1.legend(['original', 'fitted'])
    ax1.set_ylabel('voltage')
    ax1.set_xlabel('Time')

    ax2 = ax1.twinx()
    ax2.set_ylabel('voltage error')
    ax2.scatter(range(len(V_pred)),V_pred - voltage,alpha = 0.3)

    plt.title('recovered voltage')

    plt.legend(['original','fitted'])
    plt.show()

wtt = 1

if wtt == 1:
    import numpy
    a = tf.Variable(initial_value=0.005193)
    b = tf.Variable(initial_value=0.1045)
    c = tf.Variable(initial_value=1.438)
    d = tf.Variable(initial_value=-104.2)
    e =  tf.Variable(initial_value=2217.)
    f =  tf.Variable(initial_value=9.669e-05)
    g =  tf.Variable(initial_value=-3.171e-07)
    plt.plot(voltage)

    V_pred = []
    for i in range(len(T_out)):
        T= (T_out[i] + T_in[i])/2
        T2 = tf.math.square(T)

        V_pred_cur = 1.229
        V_pred_cur -= a * (T-25)
        V_pred_cur += (f + g * T)*current_density[i]
        V_pred_cur += b * tf.math.log(( c + d/T + e/T2 )*current_density[i] + 1.0)
        V_pred.append(V_pred_cur)

    plt.plot(V_pred)
    plt.title('recovered voltage by wtt')
    plt.ylabel('voltage')
    plt.xlabel('time')
    plt.legend(['original', 'fitted'])
    plt.show()


