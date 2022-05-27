'''
与之前的内容不同，这里主要希望能够使用神经网络来拟合极化曲线，使用的数据还是之前的数据
'''
import time

import numpy as np
import pandas
from matplotlib import pyplot as plt
import tensorflow as tf


def varname(var, all_var=locals()):
    return [varname for varname in all_var if all_var[varname] is var][0]


def Vres(Temp):
    '''这里就是计算热中性电压'''
    import numpy as np
    T_ref = 25
    F = 96485
    n = 2
    R = 8.3145
    CH2O = 75  # 参考点状态下的水热容(单位：J/(K*mol))
    CH2 = 29
    CO2 = 29
    S0_H2 = 131
    S0_H20 = 70
    S0_O2 = 205
    DHH2O = -2.86 * 10 ** 5 + CH2O * (Temp - T_ref)  # 参考点状态下的焓变(单位：J/mol)
    DHH2 = 0 + CH2 * (Temp - T_ref)  # 参考点状态下的焓变(单位：J/mol)
    DHO2 = 0 + CO2 * (Temp - T_ref)  # 参考点状态下的焓变(单位：J/mol)
    DH = DHH2 + DHO2 / 2 - DHH2O
    SH2 = CH2 * np.math.log((Temp + 273.15) / (T_ref + 273.15), 10) - R * np.math.log(10, 10) + S0_H2
    SO2 = CO2 * np.math.log((Temp + 273.15) / (T_ref + 273.15), 10) - R * np.math.log(10, 10) + S0_O2
    SH20 = CH2O * np.math.log((Temp + 273.15) / (T_ref + 273.15), 10) + S0_H20
    DS = SH2 + 0.5 * SO2 - SH20
    DG = DH - (Temp + 273.15) * DS
    return DG / (n * F)


def polar_lihao(Temp, current):
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
    U = Vres(Temp) + (r1 + r2 * Temp) * j + (s1 + s2 * Temp + s3 * Temp ** 2) * math.log(
        ((t1 + t2 / Temp + t3 / Temp ** 2) * j + 1))
    return U


def retrive_polar_v_timesequence(T_out_seq, current, n_cell=34):
    re_V = []
    for i in range(len(T_out_seq)):
        re_V.append(polar_lihao(T_out_seq[i], current[i]) * n_cell)
    return re_V


def compare_polar_and_original(V_res, voltage):
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
    ax2.plot([0] * len(error), color='r', alpha=0.7)
    ax2.scatter(range(len(error)), error, alpha=0.3)
    plt.title('recovered voltage-neural networks')
    plt.legend(['error'], loc=9)
    plt.subplots_adjust(left=0.057, bottom=0.062, right=0.95, top=0.95)
    plt.show()


class PolarCurve(tf.keras.Model):
    '''定义一个model类，来进行神经网络的传输和训练'''

    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(units=30)
        self.dense2 = tf.keras.layers.Dense(units=60, activation='sigmoid')
        self.dense3 = tf.keras.layers.Dense(units=60, activation='sigmoid')
        self.dense4 = tf.keras.layers.Dense(units=30)
        self.dense_end = tf.keras.layers.Dense(units=1)

    def call(self, inputs):
        # T_out, T_in, current_density, lam = zip(*inputs)
        # T_out = np.array(T_out)
        # T_in = np.array(T_in)
        # print(np.concatenate(T_out,T_in))
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense_end(x)
        return x


class DataLoader():
    '''通过一个类来读取数据，并且随机给到batch'''

    def __init__(self):
        import pandas
        import numpy as np
        SourceFile = '20s/Polar Fitting/All Polar-like data.csv'  # 和0420版本的主要区别就在于更换了数据
        df = pandas.read_csv(SourceFile)
        self.length = len(df)

        voltage = np.array(df['Voltage']).reshape(len(df), 1)
        current = np.array(df['Current']).reshape(len(df), 1)
        T_in = np.array(df['T_in']).reshape(len(df), 1)
        T_out = np.array(df['T_out']).reshape(len(df), 1)
        LyeFlow = np.array(df['LyeFlow']).reshape(len(df), 1)

        self.voltage = np.array(df['Voltage']).reshape(len(df), 1) / 34
        self.current_density = np.array(df['Current']).reshape(len(df), 1) / 0.425
        self.T_out = np.array(df['T_out']).reshape(len(df), 1)
        self.T_in = np.array(df['T_in']).reshape(len(df), 1)
        self.LyeFlow = np.array(df['LyeFlow']).reshape(len(df), 1)

        lam = 0.204425 / LyeFlow
        current /= 0.425 * 4000
        voltage /= 34
        T_out /= 100
        T_in /= 100

        self.all_data_x = np.concatenate((T_out, T_in, current, lam), axis=1)
        self.all_data_y = voltage

    def get_batch(self, batch_size=50):
        seq = []
        vol = []
        for i in range(batch_size):
            index = np.random.randint(0, len(self.all_data_x))
            seq.append(self.all_data_x[index])
            vol.append(self.all_data_y[index])
        return np.array(seq), np.array(vol)

    def get_polar_data(self):
        '''这部分是针对PolarCurve_shg这个函数的输入需求，返回值'''
        return self.T_out, self.T_in, self.current_density, self.LyeFlow

    def get_all_data(self):
        return self.all_data_x, self.all_data_y

    def PolarCurve_shg(self):
        LyeFlow = self.LyeFlow
        T_out = self.T_out
        T_in = self.T_in
        current_density = self.current_density
        lam = 0.204425 / LyeFlow
        '''这是极化曲线部分的参数，已经经过训练'''
        ar1 = 7.759098e-05
        ar2 = 8.173941e-06
        ar3 = -6.440736e-08
        as1 = 0.0645596
        as2 = 0.0003861618
        as3 = -2.1838548e-05
        at1 = 0.19026381
        at2 = 6.4511757
        at3 = 3700.711
        br1 = 6.150415e-05
        br2 = 3.845287e-07
        br3 = -1.4237826e-08
        bs1 = 0.06504009
        bs2 = 0.00015039518
        bs3 = 1.1256178e-05
        bt1 = 0.19018775
        bt2 = -26.770304
        bt3 = 1202.0524

        Ures = []
        T_ave = lam * T_out + (1 - lam) * T_in
        for i in range(len(T_out)):
            Ures.append(Vres(T_ave[i]))

        V_out = current_density * (ar1 + ar2 * T_out + ar3 * T_out ** 2) + (as1 + as2 * T_out + as3 * T_out ** 2) \
                * tf.math.log(current_density * (at1 + at2 / T_out + at3 / T_out ** 2) + 1)
        V_in = current_density * (br1 + br2 * T_in + br3 * T_in ** 2) + (bs1 + bs2 * T_in + bs3 * T_in ** 2) \
               * tf.math.log(current_density * (bt1 + bt2 / T_in + bt3 / T_in ** 2) + 1)
        V_pred = Ures + tf.sqrt(lam * V_out ** 2 + (1 - lam) * V_in ** 2)
        return V_pred.numpy()


t0 = time.time()
num_epochs = 0
learning_rate = 1E-3
batch_size = 50
storage_file = 'Polarization/Trial/checkpoint-0423-3.ckpt'
dataloader = DataLoader()

read_model = 1
if read_model == 1:
    model = tf.keras.models.load_model('Neural Networks/Polarization/Trial/checkpoint-0423-3.ckpt')
else:
    model = PolarCurve()

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
num_batches = int(dataloader.length // batch_size)
loss_seq = []
for epoch_id in range(num_epochs):
    loss_epoch = 0
    for batch_id in range(num_batches):
        inputs, voltage = dataloader.get_batch(batch_size=batch_size)
        with tf.GradientTape() as tape:
            vol_pred = model(inputs)
            loss = tf.reduce_sum(tf.square(vol_pred - voltage))
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))
        loss_epoch += loss.numpy()
    print('epoch %d,\t loss = %f' % (epoch_id, loss_epoch))
    loss_seq.append(loss_epoch)

save = 1
if save == 1:
    model.save('Neural Networks/Polarization/Standard-0423/standard-model.ckpt')

restore = 1
if restore == 1:
    model_restore = tf.keras.models.load_model(storage_file)
else:
    model_restore = model

pred = 1
if pred == 1:
    inputs, voltage = dataloader.get_all_data()
    vol_recover = model_restore(inputs)
    loss_total = tf.reduce_sum(tf.square(vol_recover - voltage))
    print(storage_file)
    print('Overall loss = ', loss_total.numpy())
    print('Epoches = ', num_epochs, 'time = ', time.time() - t0)
    compare_polar_and_original(vol_recover, voltage)
