import numpy as np
import tensorflow as tf

def Vres(Temp):
    """这里就是计算热中性电压"""
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


def polar_lihao(Temp, current_density):
    #polar_collection.polar_lihao((T_out+T_in)/2,current_density)#标准使用方法
    """李昊版本的极化曲线"""
    r1 = 0.0001362
    r2 = -1.316e-06
    s1 = 0.06494
    s2 = 0.0013154
    s3 = -4.296e-06
    t1 = 0.1645
    t2 = -18.96
    t3 = 672.5
    j = current_density
    if isinstance(Temp,int) or isinstance(Temp,float):
        Ures = Vres(Temp)
    else:
        Ures = []
        for i in range(len(Temp)):
            Ures.append(Vres(Temp[i]))
    U = Ures + (r1 + r2 * Temp) * j + (s1 + s2 * Temp + s3 * Temp ** 2) * np.log(
        ((t1 + t2 / Temp + t3 / Temp ** 2) * j + 1))
    return U


def polar_wtt(T_out,  current_density):
    #polar_collection.polar_wtt(T_out,T_in,current_density)#标准引用格式
    """王恬恬版本的极化曲线"""
    a = 0.005193
    b = 0.1045
    c = 1.438
    d = -104.2
    e = 2217
    f = 9.669e-05
    g = -3.171e-07
    V_pred = []
    if isinstance(T_out,float) or isinstance(T_out,int):
        V_pred_cur = 1.229
        V_pred_cur -= a * (T_out - 25)
        V_pred_cur += b * np.log((c + d / T_out + e / T_out ** 2) * current_density + 1)
        V_pred_cur += (f + g * T_out) * current_density
        V_pred= V_pred_cur

    else:
        for i in range(len(T_out)):
            V_pred_cur = 1.229
            V_pred_cur -= a * (T_out[i] - 25)
            V_pred_cur += b * np.log((c + d / T_out[i] + e / T_out[i] ** 2) * current_density[i] + 1)
            V_pred_cur += (f + g * T_out[i]) * current_density[i]
            V_pred.append(V_pred_cur)
    return V_pred

def polar_shg(T_out,T_in,current_density,LyeFlow):
    #polar_collection.polar_shg(T_out,T_in,current_density,LyeFlow)#标准引用格式
    lam = 0.204425 / LyeFlow
    '''这是极化曲线部分的参数，已经经过训练'''
    ar1 = 7.39445e-05
    ar2 = 5.759043e-06
    ar3 = -5.5199894e-08
    as1 = 0.06491715
    as2 = 0.00036700224
    as3 = -1.561482e-05
    at1 = 0.19111753
    at2 = -47.666733
    at3 = 3657.258
    br1 = 6.4338914e-05
    br2 = 5.231239e-07
    br3 = -1.3664759e-08
    bs1 = 0.06475956
    bs2 = 0.0001710047
    bs3 = 9.757855e-06
    bt1 = 0.19021238
    bt2 = -26.661955
    bt3 = 1217.563
    if isinstance(T_out,int) or isinstance(T_out,float):
        T_ave = lam * T_out + (1 - lam) * T_in
        Ures = Vres(T_ave)
    else:
        Ures = []
        T_ave = lam * T_out + (1 - lam) * T_in
        for i in range(len(T_out)):
            Ures.append(Vres(T_ave[i]))

    V_out = current_density * (ar1 + ar2 * T_out + ar3 * T_out ** 2) + (as1 + as2 * T_out + as3 * T_out ** 2) \
            * np.log(current_density * (at1 + at2 / T_out + at3 / T_out ** 2) + 1)
    V_in = current_density * (br1 + br2 * T_in + br3 * T_in ** 2) + (bs1 + bs2 * T_in + bs3 * T_in ** 2) \
           * np.log(current_density * (bt1 + bt2 / T_in + bt3 / T_in ** 2) + 1)
    V_pred = Ures + lam * V_out + (1-lam) * V_in
    return V_pred

def compare_polar_and_original(V_res, voltage):
    from matplotlib import pyplot as plt
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


class polar_nn():
    """可以直接定义一个类，可以直接加载神经网络，给出结果"""
    def __init__(self,storage_file = 'Neural Networks/Polarization/Standard-0424/standard-model.ckpt'):
        self.model = tf.keras.models.load_model(storage_file)

    def predict(self,inputs):
        return self.model(inputs)

    def polar(self,T_out,T_in,current_density,LyeFlow):
        T_out = self.exp_dims(T_out)
        T_in = self.exp_dims(T_in)
        current_density = self.exp_dims(current_density)
        LyeFlow = self.exp_dims(LyeFlow)
        inputs = np.concatenate([T_out/100, T_in/100, current_density/4000, 0.204425 / LyeFlow], axis=1)
        threshold = 750.#此界限以上是用神经网络，此阈值一下使用极化曲线公式

        output = []
        for i in range(len(inputs)):
            """因为以后可能需要我们根据500电密以下的进行两个极化曲线的拼接，所以说需要在这里准备好逐点进行极化特性输出"""
            if current_density[i] >=threshold:
                output.append(float(self.predict(inputs= inputs[i:i+1])))
            elif current_density[i] <= 21.5:
                output.append(float(Vres((T_out[i] + T_in[i])/2)))
            else:
                input_cur = np.array([[float(inputs[i][0]),float(inputs[i][1]),threshold/4000,float(inputs[i][3])]])
                std_nn = float(self.predict(inputs= input_cur))
                std_shg = float(polar_shg(float(inputs[i][0])*100,float(inputs[i][1])*100,threshold,0.204425/float(inputs[i][3])))
                ratio = std_nn/std_shg
                output.append(ratio * float(polar_shg(float(inputs[i][0])*100,float(inputs[i][1])*100,float(inputs[i][2])*4000,0.204425/float(inputs[i][3]))))
        return output

    def exp_dims(self,input):
        if isinstance(input,int) or isinstance(input,float) :
            input = np.array([input])
        if len(input.shape) ==1:
            return np.expand_dims(input,1)
        else:
            return input



class polar_nn_shg_trainable(tf.keras.Model):
    """定义一个model类，来进行神经网络的传输和训练"""

    def __init__(self):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(units=10)
        self.dense2 = tf.keras.layers.Dense(units=30, activation='sigmoid')
        self.dense3 = tf.keras.layers.Dense(units=40, activation='sigmoid')
        self.dense4 = tf.keras.layers.Dense(units=20)
        self.dense5 = tf.keras.layers.Dense(units=30, activation='sigmoid')
        self.dense_end = tf.keras.layers.Dense(units=1)

    def call(self, inputs):
        """一开始只输入入口温度、出口温度、碱液流量比，最后在让它和电流密度相乘"""
        Temp_inputs = tf.concat(
            [tf.expand_dims(inputs[:, 0], 1), tf.expand_dims(inputs[:, 1], 1), tf.expand_dims(inputs[:, 3], 1)], 1)
        j = tf.expand_dims(inputs[:, 2], 1)
        x = self.dense1(Temp_inputs)
        x = self.dense2(x)
        j_input = self.dense5(j)  # 加入电流密度
        x = self.dense3(tf.concat([x, j_input], 1))
        x = self.dense4(x)
        j_index = self.dense_end(x)
        return j_index * j + 1.229

    def restore(self):
        # model_nn = polar_collection.polar_nn_shg()# 调用方法
        # model_nn = model_nn.restore()
        storage_file = 'Neural Networks/Polarization/Standard-0424/standard-model.ckpt'
        model = tf.keras.models.load_model(storage_file)
        return model

def error_N_analysis(error):
    """计算-0.4-0.4之间的error的正态分布情况"""
    from scipy import stats
    import numpy as np
    error = np.array(error)
    error = np.squeeze(error)
    seq = []
    for e in error:
        if -0.05<=e<=0.05:
            seq.append(e)
    a,b =  stats.norm.fit(seq)
    return a,b